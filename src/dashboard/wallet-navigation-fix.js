"use strict";

/*
 * ================================================================
 * Bitcoin Transaction Intelligence Platform
 * Wallet Navigation Integration
 * M14
 * ================================================================
 *
 * Purpose:
 *   Integrate the dynamically-created Wallet / Address workspace
 *   with the existing application-section navigation.
 *
 * IMPORTANT:
 *   Wallet content must only be visible when Wallet / Address is
 *   the selected workspace.
 *
 *   The sidebar button remains visible on every page.
 *
 * ================================================================
 */


(function initializeWalletNavigation() {


    const WALLET_SECTION_ID =
        "wallet-section";


    const WALLET_NAV_SELECTOR =
        '[data-section="wallet-section"]';


    function walletSection() {

        return document.getElementById(
            WALLET_SECTION_ID
        );

    }


    function walletButton() {

        return document.querySelector(
            WALLET_NAV_SELECTOR
        );

    }


    function allNavigationButtons() {

        return Array.from(
            document.querySelectorAll(
                ".nav-button"
            )
        );

    }


    function allApplicationSections() {

        return Array.from(
            document.querySelectorAll(
                ".application-section"
            )
        );

    }


    function hideWallet() {

        const section =
            walletSection();


        if (!section) {
            return;
        }


        section.hidden =
            true;


        section.classList.remove(
            "active-section"
        );


        section.style.display =
            "none";

    }


    function showWallet() {

        const section =
            walletSection();


        if (!section) {
            return;
        }


        /*
         * Hide every normal dashboard workspace.
         */

        allApplicationSections()
            .forEach(
                currentSection => {

                    currentSection.classList.remove(
                        "active-section"
                    );

                    currentSection.hidden =
                        true;

                }
            );


        /*
         * Show Wallet workspace.
         */

        section.hidden =
            false;


        section.classList.add(
            "active-section"
        );


        section.style.display =
            "block";


        /*
         * Synchronize sidebar state.
         */

        allNavigationButtons()
            .forEach(
                button => {

                    button.classList.toggle(
                        "active",
                        button.dataset.section ===
                        WALLET_SECTION_ID
                    );

                }
            );


        window.scrollTo(
            {
                top: 0,
                behavior: "smooth"
            }
        );

    }


    function handleNavigation(
        event
    ) {

        const button =
            event.target.closest(
                ".nav-button"
            );


        if (!button) {
            return;
        }


        const target =
            button.dataset.section;


        /*
         * Wallet selected.
         *
         * app.js may also receive this click. We wait one
         * event-loop cycle so the existing navigation finishes,
         * then explicitly activate the wallet workspace.
         */

        if (
            target ===
            WALLET_SECTION_ID
        ) {

            window.setTimeout(
                showWallet,
                0
            );

            return;
        }


        /*
         * Any other workspace selected.
         *
         * The dynamically-created wallet is NOT part of
         * app.js's ".application-section" collection, so
         * we must explicitly hide it.
         */

        hideWallet();

    }


    function initialiseWalletState() {

        const section =
            walletSection();


        if (!section) {
            return;
        }


        /*
         * Wallet must start hidden.
         */

        hideWallet();


        /*
         * Make sure the sidebar button itself exists.
         */

        const button =
            walletButton();


        if (button) {

            button.classList.remove(
                "active"
            );

        }

    }


    function waitForWalletWorkspace() {

        let attempts =
            0;


        const maximumAttempts =
            100;


        const timer =
            window.setInterval(
                () => {

                    attempts += 1;


                    if (
                        walletSection() &&
                        walletButton()
                    ) {

                        window.clearInterval(
                            timer
                        );


                        initialiseWalletState();


                        console.log(
                            "Wallet navigation integration ready."
                        );


                        return;
                    }


                    if (
                        attempts >=
                        maximumAttempts
                    ) {

                        window.clearInterval(
                            timer
                        );


                        console.warn(
                            "Wallet navigation integration: wallet workspace not found."
                        );

                    }

                },
                100
            );

    }


    function initialise() {

        /*
         * Event delegation is intentional because wallet.js
         * creates the Wallet button dynamically.
         */

        document.addEventListener(
            "click",
            handleNavigation,
            false
        );


        waitForWalletWorkspace();


        console.log(
            "Wallet Navigation Integration M14 loaded."
        );

    }


    if (
        document.readyState ===
        "loading"
    ) {

        document.addEventListener(
            "DOMContentLoaded",
            initialise
        );

    } else {

        initialise();

    }


})();