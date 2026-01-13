"""
Local Chromium Browser implementation using Playwright.

This module provides a local Chromium browser implementation that runs
browser instances on the local machine using Playwright.
"""

import logging
import os
from typing import Any, Dict, Optional

from playwright.async_api import Browser as PlaywrightBrowser
from playwright.async_api import Page
from .models import (
    NavigateAction, NewTabAction, SwitchTabAction, CloseTabAction,
    ClickAction, TypeAction, EvaluateAction, BackAction, ForwardAction, RefreshAction,
    GetTextAction
)

from .browser import Browser

logger = logging.getLogger(__name__)


class LocalChromiumBrowser(Browser):
    """Local Chromium browser implementation using Playwright."""

    def __init__(
        self, launch_options: Optional[Dict[str, Any]] = None, context_options: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the local Chromium browser.

        Args:
            launch_options: Chromium-specific launch options (headless, args, etc.)
            context_options: Browser context options (viewport, user agent, etc.)
        """
        super().__init__()
        self._launch_options = launch_options or {}
        self._context_options = context_options or {}
        self._default_launch_options: Dict[str, Any] = {}
        self._default_context_options: Dict[str, Any] = {}

    async def start_platform(self) -> None:
        """Initialize the local Chromium browser platform with configuration (ASYNC)."""
        # Read environment variables
        user_data_dir = os.getenv(
            "STRANDS_BROWSER_USER_DATA_DIR", os.path.join(os.path.expanduser("~"), ".browser_automation")
        )
        headless = os.getenv("STRANDS_BROWSER_HEADLESS", "false").lower() == "true"
        width = int(os.getenv("STRANDS_BROWSER_WIDTH", "1280"))
        height = int(os.getenv("STRANDS_BROWSER_HEIGHT", "800"))

        # Ensure user data directory exists
        os.makedirs(user_data_dir, exist_ok=True)

        # Build default launch options
        self._default_launch_options = {
            "headless": headless,
            "args": [f"--window-size={width},{height}"],
        }
        self._default_launch_options.update(self._launch_options)

        self._default_context_options = {"viewport": {"width": width, "height": height}}
        self._default_context_options.update(self._context_options)

    async def create_browser_session(self) -> PlaywrightBrowser:
        """Create a new local Chromium browser instance for a session."""
        if not self._playwright:
            raise RuntimeError("Playwright not initialized")
        
        # Connect to existing CDP session if configured
        if "cdp_url" in self._default_launch_options:
            cdp_url = self._default_launch_options["cdp_url"]
            logger.info(f"Connecting to existing browser via CDP: {cdp_url}")
            
            # Retry loop for socket connection (handles race condition where Electron starts slower than Python)
            import asyncio
            last_error = None
            
            for i in range(30):
                try:
                    return await self._playwright.chromium.connect_over_cdp(cdp_url)
                except Exception as e:
                    last_error = e
                    # logger.info(f"Waiting for CDP Socket... ({i+1}/30)")
                    logger.info(f"Waiting for CDP Socket... ({i+1}/30)")
                    await asyncio.sleep(0.5)
            
            # If all retries fail, raise the last error
            logger.error(f"Failed to connect to CDP at {cdp_url} after timeout.")
            raise RuntimeError(f"Failed to connect to CDP at {cdp_url} after timeout. Last error: {last_error}")

        # Handle persistent context if specified
        if self._default_launch_options.get("persistent_context"):
            persistent_user_data_dir = self._default_launch_options.get(
                "user_data_dir", os.path.join(os.path.expanduser("~"), ".browser_automation")
            )

            # For persistent context, return the context itself as it acts like a browser
            return await self._playwright.chromium.launch_persistent_context(
                user_data_dir=persistent_user_data_dir,
                **{
                    k: v
                    for k, v in self._default_launch_options.items()
                    if k not in ["persistent_context", "user_data_dir"]
                },
            )
        else:
            # Regular browser launch
            logger.debug("launching local Chromium session browser with options: %s", self._default_launch_options)
            return await self._playwright.chromium.launch(**self._default_launch_options)

    async def close_platform(self) -> None:
        """Close the local Chromium browser. No platform specific changes needed (ASYNC)."""
        pass

    async def _setup_session_from_browser(self, browser_or_context):
        """Setup session components from browser or context."""
        import asyncio
        if isinstance(browser_or_context, PlaywrightBrowser):
            # If we are connected via CDP (Electron), try to find existing pages first
            # to ensure we attach to the window that has the Preload API loaded.
            if "cdp_url" in self._default_launch_options:
                # Retry logic to wait for initial context/page (race condition mitigation)
                # Increased to 30 attempts (15 seconds) to allow for app startup
                for i in range(30):
                    if browser_or_context.contexts:
                        # Found a context, now check for pages within it
                        ctx = browser_or_context.contexts[0]
                        if ctx.pages:
                            break
                    logger.info(f"Waiting for CDP context/pages... ({i+1}/30)")
                    await asyncio.sleep(0.5)
                
                if browser_or_context.contexts:
                    session_browser = browser_or_context
                    session_context = browser_or_context.contexts[0]
                    
                    if session_context.pages:
                        logger.info(f"Searching {len(session_context.pages)} pages for Electron Shell...")
                        
                        # Find the shell page (has electron API via window.electron)
                        shell_page = None
                        for page in session_context.pages:
                            try:
                                is_shell = await page.evaluate("!!window.electron")
                                if is_shell:
                                    shell_page = page
                                    logger.info(f"Found Electron Shell at: {page.url}")
                                    break
                            except Exception:
                                continue
                        
                        if shell_page:
                            session_page = shell_page
                            logger.info("Confirmed attached page is the Electron Shell ✅")
                        else:
                            # Fallback to first page
                            session_page = session_context.pages[0]
                            logger.warning("Attached page is NOT the Electron Shell. Browser control may be limited.")
                    else:
                        logger.error("No existing pages found in Electron context after timeout.")
                        # Do NOT create a new page here as it won't have the preload script
                        # But we must return something to avoid crash, so we warn heavily.
                        logger.warning("Creating fallback page (will likely lack shell control).")
                        session_page = await session_context.new_page()
                else:
                    logger.warning("No existing contexts found in Electron CDP. Creating new context (fallback).")
                    session_browser = browser_or_context
                    session_context = await session_browser.new_context()
                    session_page = await session_context.new_page()
            else:
                # Normal non-persistent case
                session_browser = browser_or_context
                session_context = await session_browser.new_context()
                session_page = await session_context.new_page()
        else:
            # Persistent context case
            session_context = browser_or_context
            session_browser = session_context.browser
            session_page = await session_context.new_page()

        return session_browser, session_context, session_page

    def _get_all_pages(self) -> list:
        """Get all pages from all sessions."""
        pages = []
        for session in self._sessions.values():
            if session.context and session.context.pages:
                pages.extend(session.context.pages)
        return pages

    async def _get_shell_page(self) -> Optional[Page]:
        """Find the Electron Shell Page (Main Window) which exposes window.electron."""
        # Iterate through all session contexts to find the shell
        for session in self._sessions.values():
            if not session.context:
                continue
            for page in session.context.pages:
                try:
                    is_shell = await page.evaluate("!!window.electron")
                    if is_shell:
                        return page
                except Exception:
                    continue
        return None

    async def _get_active_content_page(self, shell_page: Page) -> Optional[Page]:
        """Find the Playwright Page corresponding to the Active Tab in Electron."""
        try:
            # 1. Get active tab URL from Shell
            active_tab_info = await shell_page.evaluate("window.electron.tabs.list().then(tabs => tabs.find(t => t.isActive))")
            if not active_tab_info:
                return None
            
            target_url = active_tab_info.get("url")
            
            # 2. Find matching page in all session contexts
            # We skip the shell page itself
            for session in self._sessions.values():
                if not session.context:
                    continue
                for page in session.context.pages:
                    if page == shell_page:
                        continue
                    if page.url == target_url:
                        return page
            
            return None
        except Exception as e:
            logger.error(f"Error finding active content page: {e}")
            return None

    # --- App-Aware Action Overrides ---
    # CRITICAL: These methods MUST use window.electron and NEVER fall back to page.goto()
    # Falling back would navigate the shell page, destroying the React app!

    async def _async_navigate(self, action: NavigateAction) -> Dict[str, Any]:
        """Override: Use Shell API to navigate the active tab. NEVER use page.goto()."""
        shell = await self._get_shell_page()
        if shell:
            logger.info(f"Using Shell API to navigate to {action.url}")
            try:
                # Escape single quotes in URL to prevent JS injection
                safe_url = action.url.replace("'", "\\'")
                await shell.evaluate(f"window.electron.browser.navigate('{safe_url}')")
                return {"status": "success", "content": [{"text": f"Navigated to {action.url} via App API"}]}
            except Exception as e:
                logger.error(f"Navigation via Shell API failed: {e}")
                return {"status": "error", "content": [{"text": f"Navigation failed: {e}"}]}
        
        # DO NOT fall back to page.goto() - that would destroy the shell!
        logger.error("Shell page not found - cannot navigate safely")
        return {"status": "error", "content": [{"text": "Cannot navigate: Shell page not found. Ensure the Electron app is running."}]}

    async def _async_new_tab(self, action: NewTabAction) -> Dict[str, Any]:
        """Override: Use Shell API to create a new tab."""
        shell = await self._get_shell_page()
        if shell:
            logger.info("Using Shell API to create new tab")
            try:
                await shell.evaluate("window.electron.tabs.create()")
                return {"status": "success", "content": [{"text": "Created new tab via App API"}]}
            except Exception as e:
                return {"status": "error", "content": [{"text": f"Failed to create tab: {e}"}]}
        return {"status": "error", "content": [{"text": "Cannot create tab: Shell page not found"}]}

    async def _async_switch_tab(self, action: SwitchTabAction) -> Dict[str, Any]:
        """Override: Use Shell API to switch tabs."""
        shell = await self._get_shell_page()
        if shell:
            try:
                await shell.evaluate(f"window.electron.tabs.switch('{action.tab_id}')")
                return {"status": "success", "content": [{"text": f"Switched to tab {action.tab_id} via App API"}]}
            except Exception as e:
                return {"status": "error", "content": [{"text": f"Failed to switch tab: {e}"}]}
        return {"status": "error", "content": [{"text": "Cannot switch tab: Shell page not found"}]}

    async def _async_close_tab(self, action: CloseTabAction) -> Dict[str, Any]:
        """Override: Use Shell API to close tabs."""
        shell = await self._get_shell_page()
        if shell:
            try:
                tab_id = action.tab_id or "active"
                await shell.evaluate(f"window.electron.tabs.close('{tab_id}')")
                return {"status": "success", "content": [{"text": f"Closed tab via App API"}]}
            except Exception as e:
                return {"status": "error", "content": [{"text": f"Failed to close tab: {e}"}]}
        return {"status": "error", "content": [{"text": "Cannot close tab: Shell page not found"}]}

    # Simplified overrides for interaction (Click/Type) to target active CONTENT page dynamically
    # CRITICAL: Never fall back to base class for content interactions!

    async def _async_click(self, action: ClickAction) -> Dict[str, Any]:
        shell = await self._get_shell_page()
        if not shell:
            return {"status": "error", "content": [{"text": "Cannot click: Shell page not found. Ensure the Electron app is running."}]}
        
        page = await self._get_active_content_page(shell)
        if not page:
            return {"status": "error", "content": [{"text": "Cannot click: No active content page. Navigate to a website first using the navigate action."}]}

        try:
            await page.click(action.selector)
            return {"status": "success", "content": [{"text": f"Clicked {action.selector} on active tab"}]}
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Click failed: {str(e)}"}]}

    async def _async_type(self, action: TypeAction) -> Dict[str, Any]:
        shell = await self._get_shell_page()
        if not shell:
            return {"status": "error", "content": [{"text": "Cannot type: Shell page not found. Ensure the Electron app is running."}]}
        
        page = await self._get_active_content_page(shell)
        if not page:
            return {"status": "error", "content": [{"text": "Cannot type: No active content page. Navigate to a website first using the navigate action."}]}

        try:
            await page.fill(action.selector, action.text)
            return {"status": "success", "content": [{"text": f"Typed text into {action.selector}"}]}
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Type failed: {str(e)}"}]}

    async def _async_evaluate(self, action: EvaluateAction) -> Dict[str, Any]:
        shell = await self._get_shell_page()
        if not shell:
            return {"status": "error", "content": [{"text": "Cannot evaluate: Shell page not found."}]}
        
        # Routing Logic for Evaluate
        if "electron" in action.script.lower() and "window.electron" in action.script:
            # Run on Shell for app control
            page = shell
        else:
            # Run on content page for website interaction
            page = await self._get_active_content_page(shell)
            if not page:
                # For evaluate without electronAPI, we need a content page
                return {"status": "error", "content": [{"text": "Cannot evaluate: No active content page. Navigate to a website first."}]}

        try:
            result = await page.evaluate(action.script)
            return {"status": "success", "content": [{"text": f"Evaluation result: {result}"}]}
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Evaluate failed: {str(e)}"}]}