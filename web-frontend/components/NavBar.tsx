'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ModeToggle } from '@/components/ModeToggle';
import { NavigationLink } from '@/components/NavigationLink';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Sparkles, Settings, HelpCircle, Info, Smartphone, MoreVertical, Moon, Sun, Github, Shield, Mail } from 'lucide-react';
import { cn } from '@/lib/utils';
import { navigationItems } from '@/lib/constants';
import { useTryonStore } from '@/lib/tryon-store';
import { isMobile } from '@/lib/utils/device';
import { useTheme } from 'next-themes';
import { BackendStatusIndicator } from '@/components/BackendStatusIndicator';

export default function NavBar() {
  const pathname = usePathname();
  const { activeMode, setMode, openHelp, openAbout } = useTryonStore();
  const isTryOnPage = pathname === '/try-on';
  const [isMobileDevice, setIsMobileDevice] = useState(false);
  const { theme, setTheme } = useTheme();

  useEffect(() => {
    setIsMobileDevice(isMobile());

    const handleResize = () => {
      setIsMobileDevice(isMobile());
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <nav className="w-full border-b bg-background/80 backdrop-blur-xl sticky top-0 z-50">
      <div className="container mx-auto px-3 sm:px-4">
        <div className="flex items-center justify-between h-14 sm:h-16">
          {/* Logo Section */}
          <Link href="/" className="flex items-center space-x-2 sm:space-x-3 group">
            <div className="relative">
              <div className="w-8 h-8 sm:w-10 sm:h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl sm:rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-200">
                <Sparkles className="w-4 h-4 sm:w-6 sm:h-6 text-white" />
              </div>
              <div className="absolute -inset-1 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl sm:rounded-2xl opacity-0 group-hover:opacity-20 blur transition-all duration-200" />
            </div>
            <div>
              <h1 className="text-base sm:text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                AR Fashion
              </h1>
              <p className="text-[10px] sm:text-xs text-muted-foreground -mt-0.5 sm:-mt-1">Try-On Studio</p>
            </div>
          </Link>

          {/* Center: Navigation Menu */}
          <div className="hidden md:flex items-center space-x-1">
            {navigationItems.map((item) => {
              const isActive = pathname === item.href;
              return <NavigationLink key={item.href} item={item} isActive={isActive} />;
            })}
          </div>

          {/* Mode Switcher pill on Try-On page (Desktop & Tablet) */}
          {isTryOnPage && (
            <div className="hidden sm:flex items-center px-2">
              <Tabs
                value={activeMode}
                onValueChange={(value) => {
                  if (value === 'ar' && isMobileDevice) return;
                  setMode(value as 'ar' | 'photo');
                }}
                className="w-auto"
              >
                <TabsList className="h-9 bg-muted/60 p-1 border">
                  <TabsTrigger
                    value="ar"
                    disabled={isMobileDevice}
                    className="text-xs px-3 py-1 font-medium"
                  >
                    <span className="flex items-center gap-1.5">
                      <span>Live AR</span>
                      {isMobileDevice && (
                        <Badge variant="secondary" className="px-1 py-0 text-[9px]">
                          <Smartphone className="h-2 w-2" />
                        </Badge>
                      )}
                    </span>
                  </TabsTrigger>
                  <TabsTrigger value="photo" className="text-xs px-3 py-1 font-medium">
                    Photo HD
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
          )}

          {/* Mobile Navigation Dropdown Menu (Always visible on mobile) */}
          <div className="md:hidden">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm" className="h-8 w-8 p-0" aria-label="Navigation Menu">
                  <div className="w-4 h-4 flex flex-col justify-center space-y-1">
                    <div className="w-full h-0.5 bg-current rounded-full" />
                    <div className="w-full h-0.5 bg-current rounded-full" />
                    <div className="w-full h-0.5 bg-current rounded-full" />
                  </div>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                {navigationItems.map((item) => {
                  const isActive = pathname === item.href;
                  const Icon = item.icon;

                  return (
                    <DropdownMenuItem key={item.href} asChild>
                      <Link
                        href={item.href}
                        className={cn(
                          'flex items-center gap-3 cursor-pointer',
                          isActive && 'bg-accent'
                        )}
                      >
                        <Icon className="w-4 h-4 flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1.5">
                            <span className="font-medium text-sm truncate">{item.title}</span>
                            {item.badge && (
                              <Badge variant="secondary" className="h-3.5 px-1 text-[10px] flex-shrink-0">
                                {item.badge}
                              </Badge>
                            )}
                          </div>
                          <p className="text-xs text-muted-foreground truncate">{item.description}</p>
                        </div>
                      </Link>
                    </DropdownMenuItem>
                  );
                })}
                <DropdownMenuSeparator />
                {isTryOnPage && (
                  <>
                    <DropdownMenuItem onClick={openHelp}>
                      <HelpCircle className="w-4 h-4 mr-2" />
                      <span>Help</span>
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={openAbout}>
                      <Info className="w-4 h-4 mr-2" />
                      <span>About</span>
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                  </>
                )}
                <DropdownMenuItem asChild>
                  <Link href="/privacy" className="flex items-center gap-3 cursor-pointer">
                    <Shield className="w-4 h-4 flex-shrink-0" />
                    <span className="text-sm">Privacy</span>
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/contact" className="flex items-center gap-3 cursor-pointer">
                    <Mail className="w-4 h-4 flex-shrink-0" />
                    <span className="text-sm">Contact</span>
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <a
                    href="https://github.com/nawodyaishan/ar-fashion-tryon"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-3 cursor-pointer"
                  >
                    <Github className="w-4 h-4 flex-shrink-0" />
                    <span className="text-sm">Source Code</span>
                  </a>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>


          {/* Right Section */}
          <div className="flex items-center space-x-1 sm:space-x-2">
            {/* Try-On Page Actions - Desktop */}
            {isTryOnPage && (
              <>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={openHelp}
                  aria-label="Help"
                  className="hidden md:flex h-8 w-8 sm:h-9 sm:w-auto sm:px-3"
                >
                  <HelpCircle className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                  <span className="ml-2 hidden lg:inline">Help</span>
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={openAbout}
                  aria-label="About"
                  className="hidden md:flex h-8 w-8 sm:h-9 sm:w-auto sm:px-3"
                >
                  <Info className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                  <span className="ml-2 hidden lg:inline">About</span>
                </Button>
              </>
            )}

            {/* Settings Button - Desktop (not on try-on page) */}
            {!isTryOnPage && (
              <div className="hidden md:block">
                <Link href="/settings">
                  <Button
                    variant={pathname === '/settings' ? 'default' : 'outline'}
                    size="sm"
                    className="h-8 sm:h-10"
                  >
                    <Settings className="h-3.5 w-3.5 sm:h-4 sm:w-4 sm:mr-2" />
                    <span className="hidden sm:inline">Settings</span>
                  </Button>
                </Link>
              </div>
            )}

            {/* Backend Status Indicator */}
            {!isTryOnPage && (
              <div className="hidden lg:block">
                <BackendStatusIndicator />
              </div>
            )}

            {/* Footer Links - Desktop Only */}
            {!isTryOnPage && (
              <div className="hidden lg:flex items-center gap-1">
                <Link href="/privacy">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 px-2"
                    title="Privacy Policy"
                  >
                    <Shield className="h-3.5 w-3.5" />
                  </Button>
                </Link>
                <Link href="/contact">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 px-2"
                    title="Contact"
                  >
                    <Mail className="h-3.5 w-3.5" />
                  </Button>
                </Link>
                <a href="https://github.com/nawodyaishan/ar-fashion-tryon" target="_blank" rel="noopener noreferrer">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 px-2"
                    title="Source Code"
                  >
                    <Github className="h-3.5 w-3.5" />
                  </Button>
                </a>
              </div>
            )}

            {/* Desktop Theme Toggle */}
            <div className="hidden md:block">
              <ModeToggle />
            </div>

            {/* Mobile Dropdown Menu */}
            {isTryOnPage && (
              <div className="md:hidden">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 w-8 p-0"
                      aria-label="Menu"
                    >
                      <MoreVertical className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-48">
                    <DropdownMenuItem onClick={openHelp}>
                      <HelpCircle className="mr-2 h-4 w-4" />
                      <span>Help</span>
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={openAbout}>
                      <Info className="mr-2 h-4 w-4" />
                      <span>About</span>
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
                      {theme === 'dark' ? (
                        <>
                          <Sun className="mr-2 h-4 w-4" />
                          <span>Light Mode</span>
                        </>
                      ) : (
                        <>
                          <Moon className="mr-2 h-4 w-4" />
                          <span>Dark Mode</span>
                        </>
                      )}
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
