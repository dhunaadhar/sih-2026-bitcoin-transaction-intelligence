"use strict";

/*
 * Bitcoin Transaction Intelligence Platform
 * M14.6 - Wallet / Address Investigation Workspace
 *
 * This module is intentionally isolated from app.js.
 * It uses the existing offline API:
 *
 *   GET /api/wallets/search?q=<query>
 *   GET /api/wallet/<address>
 *
 * It does not introduce external libraries or network calls.
 */


// ================================================================
// WALLET WORKSPACE
// ================================================================

(function initializeWalletWorkspace() {

    const API_BASE_WALLET = "/api";


    function escapeWalletHtml(value) {

        if (typeof escapeHtml === "function") {
            return escapeHtml(value);
        }

        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }


    function walletNumber(value, digits = 4) {

        const number = Number(value);

        if (!Number.isFinite(number)) {
            return "—";
        }

        return number.toLocaleString(
            undefined,
            {
                maximumFractionDigits: digits
            }
        );
    }


    function walletText(value) {

        if (
            value === null ||
            value === undefined ||
            value === ""
        ) {
            return "—";
        }

        return String(value);
    }


    async function walletFetch(path) {

        const response =
            await fetch(
                `${API_BASE_WALLET}${path}`,
                {
                    method: "GET",
                    headers: {
                        "Accept": "application/json"
                    }
                }
            );


        let data = null;

        try {
            data = await response.json();
        } catch {
            data = null;
        }


        if (!response.ok) {

            const detail =
                data &&
                data.detail
                    ? data.detail
                    : `Request failed (${response.status})`;

            throw new Error(detail);
        }


        return data;
    }


    // ============================================================
    // DYNAMIC WORKSPACE
    // ============================================================

    function createNavigationButton() {

        const navigation =
            document.getElementById(
                "main-navigation"
            );

        if (!navigation) {
            return;
        }


        if (
            document.querySelector(
                '[data-section="wallet-section"]'
            )
        ) {
            return;
        }


        const button =
            document.createElement(
                "button"
            );

        button.className =
            "nav-button";

        button.dataset.section =
            "wallet-section";

        button.textContent =
            "Wallet / Address";


        navigation.appendChild(
            button
        );
    }


    function createWorkspace() {

        if (
            document.getElementById(
                "wallet-section"
            )
        ) {
            return;
        }


        const main =
            document.querySelector(
                "main"
            );

        if (!main) {
            return;
        }


        const section =
            document.createElement(
                "section"
            );

        section.id =
            "wallet-section";

        section.className =
            "dashboard-section section";

        section.hidden =
            true;


        section.innerHTML =
            `
            <div class="section-heading">

                <div>

                    <div class="section-kicker">
                        ENTITY INVESTIGATION
                    </div>

                    <h2>
                        Wallet / Address Intelligence
                    </h2>

                    <p class="section-description">
                        Search a Bitcoin address and inspect its
                        observable transaction relationships in the
                        offline investigation graph.
                    </p>

                </div>

                <div class="offline-badge">
                    OFFLINE GRAPH
                </div>

            </div>


            <div
                class="panel-card"
                style="
                    margin-bottom:18px;
                "
            >

                <div class="panel-title">
                    Address Lookup
                </div>

                <div
                    class="search-row"
                    style="
                        margin-top:14px;
                    "
                >

                    <input
                        id="wallet-search-input"
                        type="text"
                        placeholder="Enter full Bitcoin address or address prefix"
                        autocomplete="off"
                        spellcheck="false"
                    >

                    <button
                        id="wallet-search-button"
                        class="button primary"
                    >
                        Search Address
                    </button>

                </div>


                <div
                    id="wallet-search-message"
                    class="form-message"
                ></div>


                <div
                    id="wallet-search-results"
                    style="
                        margin-top:16px;
                    "
                ></div>

            </div>


            <div
                id="wallet-investigation-empty"
                class="empty-panel"
            >
                Search for a wallet/address to begin
                entity investigation.
            </div>


            <div
                id="wallet-investigation-panel"
                hidden
            >

                <div
                    class="metric-grid"
                    style="
                        margin-bottom:18px;
                    "
                >

                    <article class="metric-card">

                        <div class="metric-label">
                            Address
                        </div>

                        <div
                            id="wallet-address-stat"
                            class="metric-value mono"
                            style="
                                font-size:15px;
                                word-break:break-all;
                            "
                        >
                            —
                        </div>

                        <div class="metric-detail">
                            Investigated wallet
                        </div>

                    </article>


                    <article class="metric-card">

                        <div class="metric-label">
                            Transactions
                        </div>

                        <div
                            id="wallet-transaction-count"
                            class="metric-value"
                        >
                            —
                        </div>

                        <div class="metric-detail">
                            Connected transaction nodes
                        </div>

                    </article>


                    <article class="metric-card">

                        <div class="metric-label">
                            Relationships
                        </div>

                        <div
                            id="wallet-relationship-count"
                            class="metric-value"
                        >
                            —
                        </div>

                        <div class="metric-detail">
                            Observable graph edges
                        </div>

                    </article>


                    <article class="metric-card">

                        <div class="metric-label">
                            Transaction Risk
                        </div>

                        <div
                            id="wallet-risk-summary"
                            class="metric-value"
                        >
                            —
                        </div>

                        <div class="metric-detail">
                            Maximum connected transaction risk
                        </div>

                    </article>

                </div>


                <div
                    class="workspace-grid"
                    style="
                        grid-template-columns:
                            minmax(0,1fr)
                            minmax(0,1fr);
                        margin-bottom:18px;
                    "
                >

                    <div class="panel-card">

                        <div class="panel-title">
                            Address Profile
                        </div>

                        <div
                            id="wallet-profile"
                            class="detail-list"
                        ></div>

                    </div>


                    <div class="panel-card">

                        <div class="panel-title">
                            Investigation Actions
                        </div>

                        <div
                            style="
                                display:grid;
                                gap:10px;
                                margin-top:14px;
                            "
                        >

                            <button
                                id="wallet-open-graph"
                                class="button primary"
                            >
                                Open Wallet Graph
                            </button>

                            <button
                                id="wallet-clear"
                                class="button secondary"
                            >
                                Clear Investigation
                            </button>

                        </div>


                        <div
                            id="wallet-action-message"
                            class="form-message"
                        ></div>

                    </div>

                </div>


                <div class="panel-card">

                    <div class="panel-title">
                        Connected Transactions
                    </div>

                    <p class="panel-description">
                        Transactions directly connected to this
                        wallet through the unified investigation graph.
                        Select a transaction to open its full
                        transaction investigation.
                    </p>


                    <div
                        class="table-container"
                        style="
                            margin-top:14px;
                            max-height:520px;
                            overflow:auto;
                        "
                    >

                        <table>

                            <thead>

                                <tr>

                                    <th>
                                        TXID
                                    </th>

                                    <th>
                                        Time
                                    </th>

                                    <th>
                                        Risk
                                    </th>

                                    <th>
                                        Role
                                    </th>

                                    <th>
                                        Amount
                                    </th>

                                    <th>
                                        Action
                                    </th>

                                </tr>

                            </thead>


                            <tbody
                                id="wallet-transactions-body"
                            >

                                <tr>
                                    <td
                                        colspan="6"
                                        class="empty-cell"
                                    >
                                        No transactions loaded.
                                    </td>
                                </tr>

                            </tbody>

                        </table>

                    </div>

                    <div
                        id="wallet-transaction-limit"
                        class="form-message"
                    ></div>

                </div>


                <div
                    class="workspace-grid"
                    style="
                        grid-template-columns:
                            minmax(0,1fr)
                            minmax(0,1fr);
                        margin-top:18px;
                    "
                >

                    <div class="panel-card">

                        <div class="panel-title">
                            Graph Relationships
                        </div>

                        <div
                            id="wallet-relationships"
                            class="detail-list"
                        ></div>

                    </div>


                    <div class="panel-card">

                        <div class="panel-title">
                            Observable Intelligence
                        </div>

                        <div
                            id="wallet-intelligence"
                            class="detail-list"
                        ></div>

                    </div>

                </div>


                <section
                    class="notice"
                    style="
                        margin-top:18px;
                    "
                >

                    <div class="notice-title">
                        Wallet investigation limitation
                    </div>

                    <p>
                        A wallet address is a pseudonymous blockchain
                        identifier. Its graph relationships and
                        transaction behavior do not establish the
                        real-world identity, ownership, intent or
                        criminality of an individual or organization.
                    </p>

                    <p>
                        The displayed information represents observable
                        relationships in the local investigation graph
                        and should be combined with independent
                        investigative evidence.
                    </p>

                </section>

            </div>
            `;


        main.appendChild(
            section
        );
    }


    // ============================================================
    // NAVIGATION
    // ============================================================

    function activateNavigation() {

        const button =
            document.querySelector(
                '[data-section="wallet-section"]'
            );

        if (!button) {
            return;
        }


        button.addEventListener(
            "click",
            () => {

                if (
                    typeof navigateToSection ===
                    "function"
                ) {

                    navigateToSection(
                        "wallet-section"
                    );

                } else {

                    document
                        .querySelectorAll(
                            "main > section"
                        )
                        .forEach(
                            section => {
                                section.hidden =
                                    section.id !==
                                    "wallet-section";
                            }
                        );
                }


                const input =
                    document.getElementById(
                        "wallet-search-input"
                    );

                input?.focus();
            }
        );
    }


    // ============================================================
    // SEARCH
    // ============================================================

    async function searchWallets() {

        const input =
            document.getElementById(
                "wallet-search-input"
            );

        const results =
            document.getElementById(
                "wallet-search-results"
            );

        const message =
            document.getElementById(
                "wallet-search-message"
            );


        const query =
            String(
                input?.value || ""
            ).trim();


        if (!query) {

            if (message) {
                message.textContent =
                    "Enter a Bitcoin address or address prefix.";
            }

            if (results) {
                results.innerHTML = "";
            }

            return;
        }


        if (message) {
            message.textContent =
                "Searching local wallet graph...";
        }


        if (results) {

            results.innerHTML =
                `
                <div class="empty-panel">
                    Searching offline graph...
                </div>
                `;
        }


        try {

            const data =
                await walletFetch(
                    `/wallets/search?q=${encodeURIComponent(
                        query
                    )}`
                );


            const matches =
                Array.isArray(data)
                    ? data
                    : (
                        data.matches ||
                        data.results ||
                        data.wallets ||
                        []
                    );


            renderSearchResults(
                matches
            );


            if (message) {

                message.textContent =
                    `${matches.length} address match${
                        matches.length === 1
                            ? ""
                            : "es"
                    } found.`;
            }


        } catch (error) {

            console.error(
                "Wallet search error:",
                error
            );


            if (message) {

                message.textContent =
                    `Wallet search failed: ${error.message}`;
            }


            if (results) {

                results.innerHTML =
                    `
                    <div class="empty-panel">
                        No wallet search result could be loaded.
                    </div>
                    `;
            }
        }
    }


    function renderSearchResults(
        matches
    ) {

        const container =
            document.getElementById(
                "wallet-search-results"
            );


        if (!container) {
            return;
        }


        if (!matches.length) {

            container.innerHTML =
                `
                <div class="empty-panel">
                    No matching wallet address was found
                    in the local investigation graph.
                </div>
                `;

            return;
        }


        const limited =
            matches.slice(
                0,
                25
            );


        container.innerHTML =
            `
            <div
                class="panel-card"
                style="
                    padding:0;
                    overflow:hidden;
                "
            >

                <div
                    style="
                        padding:14px 16px;
                        border-bottom:1px solid var(--border-color);
                    "
                >

                    <strong>
                        Matching Addresses
                    </strong>

                    <span
                        class="metric-detail"
                        style="margin-left:8px;"
                    >
                        Showing up to 25 matches
                    </span>

                </div>


                <div
                    style="
                        max-height:330px;
                        overflow:auto;
                    "
                >

                    <table>

                        <thead>

                            <tr>

                                <th>
                                    Address
                                </th>

                                <th>
                                    Action
                                </th>

                            </tr>

                        </thead>

                        <tbody>

                            ${
                                limited
                                    .map(
                                        match => {

                                            const address =
                                                typeof match ===
                                                "string"
                                                    ? match
                                                    : (
                                                        match.address ||
                                                        match.wallet ||
                                                        match.node_id ||
                                                        ""
                                                    );

                                            const clean =
                                                String(address)
                                                    .replace(
                                                        /^wallet:/,
                                                        ""
                                                    );


                                            if (!clean) {
                                                return "";
                                            }


                                            return `
                                            <tr>

                                                <td>
                                                    <span
                                                        class="mono"
                                                        style="
                                                            word-break:
                                                                break-all;
                                                        "
                                                    >
                                                        ${escapeWalletHtml(
                                                            clean
                                                        )}
                                                    </span>
                                                </td>

                                                <td>

                                                    <button
                                                        class="small-button wallet-inspect-button"
                                                        data-address="${escapeWalletHtml(
                                                            clean
                                                        )}"
                                                    >
                                                        Inspect
                                                    </button>

                                                </td>

                                            </tr>
                                            `;
                                        }
                                    )
                                    .join("")
                            }

                        </tbody>

                    </table>

                </div>

            </div>
            `;


        container
            .querySelectorAll(
                ".wallet-inspect-button"
            )
            .forEach(
                button => {

                    button.addEventListener(
                        "click",
                        () => {

                            inspectWallet(
                                button.dataset.address
                            );
                        }
                    );
                }
            );
    }


    // ============================================================
    // WALLET INVESTIGATION
    // ============================================================

    async function inspectWallet(
        address
    ) {

        const clean =
            String(
                address || ""
            ).trim();


        if (!clean) {
            return;
        }


        const empty =
            document.getElementById(
                "wallet-investigation-empty"
            );

        const panel =
            document.getElementById(
                "wallet-investigation-panel"
            );


        if (empty) {
            empty.hidden = true;
        }

        if (panel) {
            panel.hidden = false;
        }


        if (
            typeof navigateToSection ===
            "function"
        ) {

            navigateToSection(
                "wallet-section"
            );
        }


        setWalletText(
            "wallet-address-stat",
            clean
        );

        setWalletText(
            "wallet-transaction-count",
            "Loading..."
        );

        setWalletText(
            "wallet-relationship-count",
            "Loading..."
        );

        setWalletText(
            "wallet-risk-summary",
            "Loading..."
        );


        setWalletHtml(
            "wallet-profile",
            `<div class="loading-cell">
                Loading wallet profile...
            </div>`
        );

        setWalletHtml(
            "wallet-transactions-body",
            `
            <tr>
                <td
                    colspan="6"
                    class="empty-cell"
                >
                    Loading connected transactions...
                </td>
            </tr>
            `
        );

        setWalletHtml(
            "wallet-relationships",
            `<div class="loading-cell">
                Loading graph relationships...
            </div>`
        );

        setWalletHtml(
            "wallet-intelligence",
            `<div class="loading-cell">
                Building observable wallet summary...
            </div>`
        );


        try {

            const data =
                await walletFetch(
                    `/wallet/${encodeURIComponent(
                        clean
                    )}`
                );


            renderWalletInvestigation(
                data
            );


        } catch (error) {

            console.error(
                "Wallet investigation error:",
                error
            );


            setWalletText(
                "wallet-transaction-count",
                "—"
            );

            setWalletText(
                "wallet-relationship-count",
                "—"
            );

            setWalletText(
                "wallet-risk-summary",
                "—"
            );


            setWalletHtml(
                "wallet-profile",
                `
                <div class="error-cell">
                    Wallet investigation failed:
                    ${escapeWalletHtml(
                        error.message
                    )}
                </div>
                `
            );
        }
    }


    function renderWalletInvestigation(
        data
    ) {

        const address =
            data.wallet ||
            data.address ||
            "";


        const transactions =
            Array.isArray(
                data.transactions
            )
                ? data.transactions
                : [];


        const relationships =
            Array.isArray(
                data.relationships
            )
                ? data.relationships
                : [];


        const risks =
            transactions
                .map(
                    transaction =>
                        Number(
                            transaction.risk_score
                        )
                )
                .filter(
                    value =>
                        Number.isFinite(value)
                );


        const maxRisk =
            risks.length
                ? Math.max(...risks)
                : null;


        setWalletText(
            "wallet-address-stat",
            address
        );

        setWalletText(
            "wallet-transaction-count",
            walletNumber(
                data.transaction_count ??
                transactions.length,
                0
            )
        );

        setWalletText(
            "wallet-relationship-count",
            walletNumber(
                relationships.length,
                0
            )
        );

        setWalletText(
            "wallet-risk-summary",
            maxRisk === null
                ? "No scored tx"
                : maxRisk.toFixed(2)
        );


        renderWalletProfile(
            data,
            transactions
        );

        renderWalletTransactions(
            transactions
        );

        renderWalletRelationships(
            relationships
        );

        renderWalletIntelligence(
            transactions,
            relationships
        );


        const graphButton =
            document.getElementById(
                "wallet-open-graph"
            );


        if (graphButton) {

            graphButton.onclick =
                () => {

                    openWalletGraph(
                        address,
                        transactions
                    );
                };
        }


        const clearButton =
            document.getElementById(
                "wallet-clear"
            );


        if (clearButton) {

            clearButton.onclick =
                clearWalletInvestigation;
        }
    }


    function renderWalletProfile(
        data,
        transactions
    ) {

        const first =
            transactions.length
                ? transactions[0]
                : null;


        const timeSteps =
            transactions
                .map(
                    transaction =>
                        Number(
                            transaction.time_step
                        )
                )
                .filter(
                    value =>
                        Number.isFinite(value)
                );


        const firstTime =
            timeSteps.length
                ? Math.min(...timeSteps)
                : null;

        const lastTime =
            timeSteps.length
                ? Math.max(...timeSteps)
                : null;


        const html =
            `
            <div class="detail-row">
                <span>Address</span>
                <strong class="mono">
                    ${escapeWalletHtml(
                        data.wallet ||
                        data.address ||
                        "—"
                    )}
                </strong>
            </div>

            <div class="detail-row">
                <span>Connected transactions</span>
                <strong>
                    ${walletNumber(
                        transactions.length,
                        0
                    )}
                </strong>
            </div>

            <div class="detail-row">
                <span>First observed time step</span>
                <strong>
                    ${walletText(firstTime)}
                </strong>
            </div>

            <div class="detail-row">
                <span>Latest observed time step</span>
                <strong>
                    ${walletText(lastTime)}
                </strong>
            </div>

            <div class="detail-row">
                <span>Graph relationships</span>
                <strong>
                    ${walletNumber(
                        (
                            data.relationships ||
                            []
                        ).length,
                        0
                    )}
                </strong>
            </div>

            <div class="detail-row">
                <span>Graph query</span>
                <strong>
                    wallet_context
                </strong>
            </div>
            `;


        setWalletHtml(
            "wallet-profile",
            html
        );
    }


    function renderWalletTransactions(
        transactions
    ) {

        const body =
            document.getElementById(
                "wallet-transactions-body"
            );


        const limitMessage =
            document.getElementById(
                "wallet-transaction-limit"
            );


        if (!body) {
            return;
        }


        if (!transactions.length) {

            body.innerHTML =
                `
                <tr>
                    <td
                        colspan="6"
                        class="empty-cell"
                    >
                        No connected transactions found.
                    </td>
                </tr>
                `;

            if (limitMessage) {
                limitMessage.textContent = "";
            }

            return;
        }


        const MAX_ROWS =
            100;


        const visible =
            transactions
                .slice()
                .sort(
                    (
                        a,
                        b
                    ) =>
                        (
                            Number(
                                b.time_step
                            ) || 0
                        ) -
                        (
                            Number(
                                a.time_step
                            ) || 0
                        )
                )
                .slice(
                    0,
                    MAX_ROWS
                );


        body.innerHTML =
            visible
                .map(
                    transaction => {

                        const txid =
                            transaction.txid ||
                            transaction.node_id ||
                            "";


                        const normalizedTxid =
                            String(txid)
                                .replace(
                                    /^tx:/,
                                    ""
                                );


                        const risk =
                            Number(
                                transaction.risk_score
                            );


                        const amount =
                            transaction.output_btc_total;


                        return `
                        <tr>

                            <td>

                                <button
                                    class="txid-button wallet-tx-button"
                                    data-txid="${escapeWalletHtml(
                                        normalizedTxid
                                    )}"
                                >
                                    ${escapeWalletHtml(
                                        normalizedTxid
                                    )}
                                </button>

                            </td>

                            <td>
                                ${escapeWalletHtml(
                                    walletText(
                                        transaction.time_step
                                    )
                                )}
                            </td>

                            <td>
                                ${
                                    Number.isFinite(risk)
                                        ? risk.toFixed(2)
                                        : "—"
                                }
                            </td>

                            <td>
                                ${
                                    transaction.node_role ||
                                    "transaction"
                                }
                            </td>

                            <td>
                                ${
                                    Number.isFinite(
                                        Number(amount)
                                    )
                                        ? `${walletNumber(
                                            amount,
                                            8
                                        )} BTC`
                                        : "—"
                                }
                            </td>

                            <td>

                                <button
                                    class="small-button wallet-tx-investigate"
                                    data-txid="${escapeWalletHtml(
                                        normalizedTxid
                                    )}"
                                >
                                    Investigate
                                </button>

                            </td>

                        </tr>
                        `;
                    }
                )
                .join("");


        body
            .querySelectorAll(
                ".wallet-tx-button, .wallet-tx-investigate"
            )
            .forEach(
                button => {

                    button.addEventListener(
                        "click",
                        () => {

                            const txid =
                                button.dataset.txid;

                            if (
                                txid &&
                                typeof openInvestigation ===
                                "function"
                            ) {

                                openInvestigation(
                                    txid
                                );
                            }
                        }
                    );
                }
            );


        if (limitMessage) {

            limitMessage.textContent =
                transactions.length >
                MAX_ROWS
                    ? `Showing ${MAX_ROWS.toLocaleString()} of ${
                        transactions.length.toLocaleString()
                    } connected transactions.`
                    : `Showing all ${transactions.length.toLocaleString()} connected transactions.`;
        }
    }


    function renderWalletRelationships(
        relationships
    ) {

        const container =
            document.getElementById(
                "wallet-relationships"
            );


        if (!container) {
            return;
        }


        if (!relationships.length) {

            container.innerHTML =
                `
                <div class="empty-panel">
                    No graph relationships were returned.
                </div>
                `;

            return;
        }


        const counts =
            {};


        relationships.forEach(
            relationship => {

                const type =
                    relationship.edge_type ||
                    relationship.relationship ||
                    "UNKNOWN";

                counts[type] =
                    (
                        counts[type] ||
                        0
                    ) + 1;
            }
        );


        const entries =
            Object.entries(
                counts
            )
            .sort(
                (
                    a,
                    b
                ) =>
                    b[1] - a[1]
            );


        container.innerHTML =
            entries
                .map(
                    (
                        [
                            type,
                            count
                        ]
                    ) =>
                        `
                        <div class="detail-row">

                            <span>
                                ${escapeWalletHtml(
                                    type
                                )}
                            </span>

                            <strong>
                                ${walletNumber(
                                    count,
                                    0
                                )}
                            </strong>

                        </div>
                        `
                )
                .join("");
    }


    function renderWalletIntelligence(
        transactions,
        relationships
    ) {

        const container =
            document.getElementById(
                "wallet-intelligence"
            );


        if (!container) {
            return;
        }


        const risks =
            transactions
                .map(
                    transaction =>
                        Number(
                            transaction.risk_score
                        )
                )
                .filter(
                    value =>
                        Number.isFinite(value)
                );


        const anomaly =
            transactions
                .map(
                    transaction =>
                        Number(
                            transaction.anomaly_signal
                        )
                )
                .filter(
                    value =>
                        Number.isFinite(value)
                );


        const behavioral =
            transactions
                .map(
                    transaction =>
                        Number(
                            transaction.behavioral_signal
                        )
                )
                .filter(
                    value =>
                        Number.isFinite(value)
                );


        const entity =
            transactions
                .map(
                    transaction =>
                        Number(
                            transaction.entity_signal
                        )
                )
                .filter(
                    value =>
                        Number.isFinite(value)
                );


        const max =
            values =>
                values.length
                    ? Math.max(...values)
                    : null;


        const average =
            values =>
                values.length
                    ? (
                        values.reduce(
                            (
                                sum,
                                value
                            ) =>
                                sum + value,
                            0
                        ) /
                        values.length
                    )
                    : null;


        container.innerHTML =
            `
            <div class="detail-row">
                <span>Maximum connected risk</span>
                <strong>
                    ${
                        max(risks) === null
                            ? "—"
                            : max(risks).toFixed(2)
                    }
                </strong>
            </div>

            <div class="detail-row">
                <span>Mean connected risk</span>
                <strong>
                    ${
                        average(risks) === null
                            ? "—"
                            : average(risks).toFixed(2)
                    }
                </strong>
            </div>

            <div class="detail-row">
                <span>Maximum anomaly signal</span>
                <strong>
                    ${
                        max(anomaly) === null
                            ? "—"
                            : max(anomaly).toFixed(3)
                    }
                </strong>
            </div>

            <div class="detail-row">
                <span>Maximum behavioral signal</span>
                <strong>
                    ${
                        max(behavioral) === null
                            ? "—"
                            : max(behavioral).toFixed(3)
                    }
                </strong>
            </div>

            <div class="detail-row">
                <span>Maximum entity signal</span>
                <strong>
                    ${
                        max(entity) === null
                            ? "—"
                            : max(entity).toFixed(3)
                    }
                </strong>
            </div>

            <div class="detail-row">
                <span>Observable graph edges</span>
                <strong>
                    ${walletNumber(
                        relationships.length,
                        0
                    )}
                </strong>
            </div>
            `;
    }


    // ============================================================
    // GRAPH ACTION
    // ============================================================

    function openWalletGraph(
        address,
        transactions
    ) {

        if (
            !transactions ||
            !transactions.length
        ) {

            setWalletMessage(
                "wallet-action-message",
                "This wallet has no connected transaction from which to open the transaction graph."
            );

            return;
        }


        const first =
            transactions
                .slice()
                .sort(
                    (
                        a,
                        b
                    ) =>
                        (
                            Number(
                                b.risk_score
                            ) || 0
                        ) -
                        (
                            Number(
                                a.risk_score
                            ) || 0
                        )
                )[0];


        const txid =
            String(
                first.txid ||
                first.node_id ||
                ""
            )
            .replace(
                /^tx:/,
                ""
            );


        if (
            txid &&
            typeof openGraph ===
            "function"
        ) {

            openGraph(
                txid
            );

            return;
        }


        setWalletMessage(
            "wallet-action-message",
            `Wallet ${address} has connected transactions, but no graph transaction could be selected.`
        );
    }


    // ============================================================
    // CLEAR
    // ============================================================

    function clearWalletInvestigation() {

        const panel =
            document.getElementById(
                "wallet-investigation-panel"
            );

        const empty =
            document.getElementById(
                "wallet-investigation-empty"
            );


        if (panel) {
            panel.hidden = true;
        }

        if (empty) {
            empty.hidden = false;
        }


        setWalletHtml(
            "wallet-search-results",
            ""
        );

        setWalletText(
            "wallet-search-message",
            ""
        );


        const input =
            document.getElementById(
                "wallet-search-input"
            );

        if (input) {
            input.value = "";
        }
    }


    // ============================================================
    // HELPERS
    // ============================================================

    function setWalletText(
        id,
        value
    ) {

        const element =
            document.getElementById(
                id
            );

        if (element) {
            element.textContent =
                walletText(value);
        }
    }


    function setWalletHtml(
        id,
        html
    ) {

        const element =
            document.getElementById(
                id
            );

        if (element) {
            element.innerHTML =
                html;
        }
    }


    function setWalletMessage(
        id,
        message
    ) {

        setWalletText(
            id,
            message
        );
    }


    // ============================================================
    // INITIALIZATION
    // ============================================================

    function initialize() {

        createNavigationButton();

        createWorkspace();

        activateNavigation();


        const searchButton =
            document.getElementById(
                "wallet-search-button"
            );


        const searchInput =
            document.getElementById(
                "wallet-search-input"
            );


        searchButton?.addEventListener(
            "click",
            searchWallets
        );


        searchInput?.addEventListener(
            "keydown",
            event => {

                if (
                    event.key ===
                    "Enter"
                ) {

                    event.preventDefault();

                    searchWallets();
                }
            }
        );
    }


    if (
        document.readyState ===
        "loading"
    ) {

        document.addEventListener(
            "DOMContentLoaded",
            initialize
        );

    } else {

        initialize();
    }

})();