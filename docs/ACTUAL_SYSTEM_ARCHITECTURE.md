# Actual System Architecture

This document is derived from the active source code in this repository. It describes the current implementation, not the older design notes or deprecated backend experiments.

## Final architecture diagram

```mermaid
flowchart TB
    user([User]) --> browser[Web browser]

    subgraph presentation[Presentation layer — web-frontend (Next.js / React / TypeScript)]
      recommendPage[app/recommend/page.tsx]
      recommendForm[RecommendationForm + PersonImageUploader\nphoto, gender, occasion, colour, style, body type, skin tone]
      results[RecommendationGrid + GarmentCard\nranked garments and select-for-try-on]
      photoWizard[PhotoWizard\nphoto / reference / full-outfit workflows]
      arStage[ARStage + VideoPreview + GarmentOverlay]
      pose[usePoseDetection\nMediaPipe PoseLandmarker in browser]
      store[Zustand stores\nuseVtonStore and useTryonStore]
      recommendPage --> recommendForm
      recommendForm --> results
      results --> store
      photoWizard --> store
      arStage --> pose
      arStage --> store
    end

    browser --> recommendPage
    browser --> photoWizard
    browser --> arStage
    browser -. camera stream .-> arStage

    subgraph api[API / processing layer — garment-processing-api (FastAPI, port 5000)]
      routes[app.py routes\nPOST /recommend\nPOST /virtual_tryon\nPOST /classify_garment\nPOST /detect_garment_type\nPOST /construct_outfit\nPOST /classify_garment_by_url]
      profile[create_user_profile\nsave uploaded person image]
      resnet[ResNet50 feature extractor\n224 × 224 preprocessing → 2048-D L2 embedding]
      recommendation[HybridRecommendationService\nmetadata candidate filtering → FAISS cosine similarity\n→ rule scores → weighted ranking]
      classifier[TensorFlow garment classifier\nbest_clothing_model.h5 or clothing_model_final.h5\nlabels: trousers / tshirt / other]
      background[rembg background removal\nPNG conversion]
      outfit[construct_outfit_image\nvertical upper + lower composition]
      gradioClient[Gradio client\n/submit_function]
      routes --> profile --> resnet --> recommendation
      routes --> classifier
      routes --> background
      routes --> outfit
      routes --> gradioClient
    end

    recommendForm -- "multipart POST /recommend\nperson image + preferences" --> routes
    recommendation -- "ranked metadata, similarity and final scores" --> results
    photoWizard -- "POST /detect_garment_type\nPOST /construct_outfit (full-outfit mode)" --> routes
    store -- "multipart POST /virtual_tryon\nperson image + garment image + inference settings" --> routes
    routes -- "result, source and cutout URLs" --> photoWizard

    subgraph data[Data, model, and asset layer]
      userUploads[(uploads/users/\nlocal person-image files)]
      catalogue[(dataset/\ncleaned_styles.csv\nimages/{id}.jpg\nembeddings.npy + image_ids.npy\nfaiss_index.bin + faiss_metadata.pkl)]
      classModel[(models/\nTensorFlow .h5 classifier\nclass_labels.json\nmodel_config.json\nrejection_threshold.json)]
      localUploads[(uploads/\nlocal storage fallback\noriginals, cutouts, try-on results)]
      cloudinary[(Cloudinary image storage/CDN\nwhen credentials are configured)]
      userUploads --> resnet
      catalogue --> recommendation
      classModel --> classifier
      background --> localUploads
      outfit --> localUploads
    end

    profile --> userUploads
    routes -- "when configured" --> cloudinary
    routes -- "otherwise" --> localUploads
    localUploads --> routes

    subgraph vton[Photo virtual try-on inference — catvton-gradio / configured Hugging Face Space]
      gradioApp[app.py submit_function\nresize person and garment to 768 × 1024]
      autoMask[AutoMasker\nDensePose + SCHP human parsing\ncloth-agnostic mask]
      catvton[CatVTONPipeline\nStable Diffusion inpainting\nVAE + UNet + CatVTON attention checkpoint]
      generated[Generated try-on image]
      gradioApp --> autoMask --> catvton --> generated
    end

    gradioClient -- "person image, garment image, cloth type\nsteps, guidance scale, seed" --> gradioApp
    generated -- "Gradio file result" --> gradioClient
    gradioClient -- "result bytes" --> routes

    mediapipe[External MediaPipe model asset\nGoogle-hosted PoseLandmarker task] -. browser model download .-> pose
```

## What is implemented

### 1. Presentation layer

The active web application is `web-frontend/`, built with Next.js 15, React 19, TypeScript, Tailwind, and Zustand.

- The recommendation page accepts a person photo plus gender, occasion, preferred colour, preferred style, style preference, body type, and skin tone. It sends a multipart request to the FastAPI `/recommend` route.
- Results are rendered in a recommendation grid. Selecting a result transfers the original person file, garment URL, and inferred cloth type into the photo try-on Zustand store, then opens `/try-on`.
- The photo try-on wizard supports three implemented paths: normal single-garment input, reference-image input, and a full outfit made from separate upper and lower garment images.
- Live AR is a separate, browser-only preview. It uses the browser camera, a client-side MediaPipe PoseLandmarker model, canvas/DOM overlays, and manual or landmark-assisted placement. It does **not** call FastAPI or CatVTON, and it does not generate a photorealistic image.

### 2. FastAPI layer

`garment-processing-api/app.py` is the active server. CORS is configured and the API serves catalog images at `/images` and local fallback assets at `/uploads`.

- `/recommend` validates the person image, creates a user profile, extracts a feature vector, and ranks dataset garments.
- `/virtual_tryon` validates the person and garment images; optionally removes the garment background; converts both images to RGB PNG; delegates inference to Gradio; and stores the generated result.
- `/classify_garment`, `/classify_garment_by_url`, and `/detect_garment_type` provide garment classification, with the first two also producing background-removed cutouts.
- `/construct_outfit` classifies, cuts out, uploads, and vertically combines upper and lower garments. The output is then used as an `overall` garment in the photo try-on workflow.

### 3. Recommendation pipeline

The live `/recommend` route calls `recommend_hybrid_garments`, which delegates to `HybridRecommendationService`; the older in-memory `GARMENT_CATALOG` and `get_recommendations()` code is not used by that route.

1. `create_user_profile()` saves the person photo in `uploads/users/` and uses ImageNet-pretrained TensorFlow ResNet50 (without its top layer, global-average pooling) to produce a normalized 2048-dimensional person-image embedding.
2. The hybrid service loads `cleaned_styles.csv`, `faiss_index.bin`, and `faiss_metadata.pkl` lazily on the first request.
3. It creates a metadata-first candidate pool: gender and optional category are retained; occasion and recognised style rules are applied with controlled fallbacks. Preferred colour is a soft score, not a hard filter.
4. It reconstructs eligible normalized catalog vectors from the FAISS `IndexFlatIP` index and takes their dot product with the normalized person embedding, i.e. cosine similarity.
5. It calculates a final weighted score: 25% similarity, 25% occasion, 20% style, 15% colour, and 15% combined body-type/skin-tone rule suitability; it sorts descending and returns up to ten items.

The frontend turns a result ID into `/images/{id}.jpg` if the API did not provide a product image URL. These catalog images are served by FastAPI from `garment-processing-api/dataset/images/`.

### 4. Photo virtual try-on pipeline

1. The frontend posts person and garment image files, `cloth_type`, inference steps, guidance scale, and seed to `POST /virtual_tryon`.
2. FastAPI stores source images (Cloudinary when configured; `uploads/` otherwise), optionally applies `rembg` to the garment, and writes RGB PNG temporary inputs.
3. `services/gradio_service.py` invokes the configured local Gradio URL or Hugging Face Space at `/submit_function`.
4. `catvton-gradio/app.py` resizes/crops the person image and pads the garment image to 768 × 1024. It accepts a user mask when supplied; otherwise its `AutoMasker` uses DensePose and SCHP to derive a cloth-agnostic mask for `upper`, `lower`, or `overall`.
5. `CatVTONPipeline` performs Stable Diffusion inpainting with the CatVTON attention checkpoint. The result is returned through Gradio, normalized to RGB PNG by FastAPI, persisted, and returned to the browser as `result_url`.

## Component-to-code mapping

| Component | Actual code / artifact | Purpose |
| --- | --- | --- |
| Recommendation page and request handoff | `web-frontend/app/recommend/page.tsx` | Collects input, requests recommendations, transfers selected garments into the photo workflow. |
| Recommendation form and person upload | `web-frontend/components/recommend/RecommendationForm.tsx`, `PersonImageUploader.tsx` | Collects the implemented preference fields and photo. |
| Recommendation HTTP adapter | `web-frontend/lib/services/recommendationApi.ts`, `lib/config/api.ts` | Posts multipart data to `/recommend`; maps hybrid-service response data to UI cards. |
| Photo try-on UI | `web-frontend/components/tryon/PhotoWizard.tsx` | Captures/uploads images, chooses normal/reference/full-outfit workflow, renders output. |
| Photo workflow state | `web-frontend/lib/store/useVtonStore.ts` | Calls classification, outfit construction, and `/virtual_tryon`. |
| Browser AR preview | `web-frontend/components/tryon/ARStage.tsx`, `VideoPreview.tsx`, `GarmentOverlay.tsx`, `ContinuousTracker.tsx` | Local live camera preview and garment placement. |
| Browser pose detection | `web-frontend/lib/hooks/usePoseDetection.ts`, `lib/pose-utils.ts` | Loads MediaPipe PoseLandmarker from Google-hosted model assets and calculates overlay alignment. |
| Active API routes | `garment-processing-api/app.py` | Defines all active FastAPI endpoints and their orchestration. |
| User profile / embedding | `garment-processing-api/services/user_profile_service.py`, `services/resnet_extractor.py`, `models/user_profile.py` | Stores person photos locally and generates ResNet50 embeddings. |
| Hybrid recommender | `garment-processing-api/services/hybrid_recommendation_service.py`, `services/recommendation_service.py` | Metadata filtering, FAISS cosine similarity, rule scoring, and final ranking. |
| Recommendation data | `garment-processing-api/dataset/cleaned_styles.csv`, `images/`, `embeddings.npy`, `image_ids.npy`, `faiss_index.bin`, `faiss_metadata.pkl` | Clothing metadata, product images, source embeddings, and the prebuilt FAISS index. |
| Dataset preparation | `garment-processing-api/scripts/clean_dataset.py`, `generate_embeddings.py`, `build_faiss_index.py` | Offline CSV cleaning, ResNet50 garment embedding generation, and FAISS-index construction. |
| Garment classifier | `garment-processing-api/services/classifier.py`, `models/*.h5`, `models/*.json` | TensorFlow classification into trousers, tshirt, or other/unknown. |
| Image processing | `garment-processing-api/services/image_processing.py` | `rembg` cutouts, RGB/PNG normalization, and upper/lower composition. |
| Asset persistence | `garment-processing-api/services/cloudinary_service.py`, `config.py` | Cloudinary storage if configured; otherwise local `uploads/` fallback. |
| Gradio orchestration | `garment-processing-api/services/gradio_service.py` | Connects to configured local Gradio service or Hugging Face Space and downloads its image result. |
| CatVTON service | `catvton-gradio/app.py`, `model/pipeline.py`, `model/cloth_masker.py` | Gradio endpoint, auto masking, and CatVTON/Stable-Diffusion inference. |

## Storage and external services

| Resource | Current implementation |
| --- | --- |
| Clothing catalogue | Local CSV, `.npy`, FAISS binary/pickle artifacts, and local images under `garment-processing-api/dataset/`. |
| User profiles | Ephemeral request-time `UserProfile` objects plus local person-image files in `uploads/users/`; there is no user account/profile database. |
| Uploaded inputs and results | Cloudinary folders `garments/originals`, `garments/cutouts`, and `garments/tryon_results` when credentials exist; local `uploads/` fallback otherwise. |
| VTON model/checkpoints | CatVTON checkpoint from `zhengchong/CatVTON`, Stable Diffusion inpainting base model, VAE, and DensePose/SCHP checkpoints loaded by the CatVTON service. |
| External APIs/services | Cloudinary; Gradio (local server or configured Hugging Face Space); Hugging Face model download; Google-hosted MediaPipe task model in the browser. |

## Implemented vs. partial, planned, or inactive

| Status | Evidence and architectural interpretation |
| --- | --- |
| Implemented | Next.js UI, FastAPI endpoints, local dataset/FAISS artifacts, ResNet50 recommendation embedding, metadata and hybrid ranking, TensorFlow classifier, `rembg`, Cloudinary/local fallback, CatVTON Gradio inference, and browser MediaPipe AR preview. |
| Partial / environment-dependent | CatVTON only runs when the local Gradio service or a reachable configured Hugging Face Space is available. Cloudinary is optional; the API falls back to local disk. TensorFlow classifier `.h5` files are intentionally supplied separately and returns `UNKNOWN` when unavailable. |
| Incomplete integration | Recommendation responses contain metadata and scores but no explicit image URL. The frontend derives `/images/{id}.jpg`, which works only while the FastAPI catalogue image mount and matching files are available. The 2048-D query represents the whole person photo rather than an extracted person/garment region; the hybrid service comments acknowledge this limitation. |
| Inactive legacy code | `web-frontend/lib/services/vtonApi.ts` still includes `processImages()` targeting the old port-8000 `/process_images/` backend, but the live store calls `virtualTryOn()` at FastAPI port 5000. `deprecated-backends/` is explicitly not on the active runtime path. |
| Not implemented as active persistence | `docker-compose.yml` declares PostgreSQL and Redis, but the active frontend/FastAPI code has no database client, ORM, schema, or Redis usage. They must not be shown as used runtime storage. |
| Documentation-only / historical plans | `docs/Updated Approach ...md` mentions MongoDB, NestJS, and other architecture ideas. These are not implemented by the active code and are excluded from the diagram. |

## Cross-check summary

The active runtime path is:

`browser → Next.js client → FastAPI (port 5000) → configured CatVTON Gradio endpoint → FastAPI storage adapter → browser`.

The recommendation request remains entirely inside the FastAPI service after the browser submits the form: it reads local dataset artifacts, creates an in-memory profile with a locally saved photo, and returns ranked catalogue metadata. It neither uses PostgreSQL/Redis nor calls CatVTON until the user subsequently selects a garment and starts photo try-on.
