"use strict";

/*
 * Bitcoin Transaction Intelligence Platform
 * Offline investigator dashboard
 *
 * M14 frontend integration
 *
 * Backend:
 *   /api/health
 *   /api/summary
 *   /api/alerts
 *   /api/alerts/{txid}
 *   /api/temporal
 *   /api/graph/{txid}
 *   /api/import
 *
 * No external JavaScript libraries are required.
 */


// ================================================================
// CONFIGURATION
// ================================================================

const API_BASE = "/api";

const DEFAULT_PAGE_SIZE = 25;

const MAX_GRAPH_NODES = 500;

const NETWORK_INTELLIGENCE_CACHE_KEY =
    "sih_network_intelligence_cache";


// ================================================================
// APPLICATION STATE
// ================================================================

const state = {

    currentSection: "overview-section",

    currentPage: 1,

    pageSize: DEFAULT_PAGE_SIZE,

    alertFilters: {
        priority: "",
        status: "",
        timeStep: "",
        minRisk: "",
        maxRisk: ""
    },

    selectedTxid: null,

    selectedAlert: null,

    selectedInvestigation: null,

    selectedGraph: null,

    summary: null,

    health: null,

    temporal: null,

    alerts: null,

    importing: false,

    loadingInvestigation: false,

    loadingGraph: false

};


// ================================================================
// DOM HELPERS
// ================================================================

function $(id) {
    return document.getElementById(id);
}


function query(selector) {
    return document.querySelector(selector);
}


function queryAll(selector) {
    return Array.from(
        document.querySelectorAll(selector)
    );
}


function escapeHtml(value) {

    if (value === null || value === undefined) {
        return "";
    }

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


function formatNumber(value, digits = 0) {

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return number.toLocaleString(
        "en-IN",
        {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits
        }
    );
}


function formatRisk(value) {

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return number.toFixed(2);
}


function formatPercent(value, digits = 1) {

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return `${(number * 100).toFixed(digits)}%`;
}


function normalizeTxid(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return "";
    }

    return String(value)
        .trim()
        .replace(/\.0$/, "");
}


function setText(id, value) {

    const element = $(id);

    if (!element) {
        return;
    }

    element.textContent =
        value === null ||
        value === undefined ||
        value === ""
            ? "—"
            : String(value);
}


function setHtml(id, html) {

    const element = $(id);

    if (!element) {
        return;
    }

    element.innerHTML = html;
}


function show(id) {

    const element = $(id);

    if (!element) {
        return;
    }

    element.classList.remove(
        "hidden"
    );
}


function hide(id) {

    const element = $(id);

    if (!element) {
        return;
    }

    element.classList.add(
        "hidden"
    );
}


function setMessage(
    id,
    message,
    type = ""
) {

    const element = $(id);

    if (!element) {
        return;
    }

    element.textContent = message || "";

    element.classList.remove(
        "success",
        "error",
        "warning"
    );

    if (type) {
        element.classList.add(type);
    }
}


function setLoading(
    id,
    message = "Loading..."
) {

    const element = $(id);

    if (!element) {
        return;
    }

    element.innerHTML = `
        <div class="loading">
            ${escapeHtml(message)}
        </div>
    `;
}


function priorityClass(priority) {

    const normalized = String(
        priority || ""
    )
        .toLowerCase()
        .replaceAll("_", "-");

    return `priority-${normalized}`;
}


function riskClass(risk) {

    const number = Number(risk);

    if (!Number.isFinite(number)) {
        return "";
    }

    if (number >= 80) {
        return "risk-very-high";
    }

    if (number >= 60) {
        return "risk-high";
    }

    if (number >= 40) {
        return "risk-moderate";
    }

    if (number >= 20) {
        return "risk-guarded";
    }

    return "risk-low";
}


// ================================================================
// API
// ================================================================

async function apiFetch(
    path,
    options = {}
) {

    const response = await fetch(
        `${API_BASE}${path}`,
        {
            ...options,
            headers: {
                ...(options.headers || {})
            }
        }
    );

    let payload = null;

    const contentType =
        response.headers.get(
            "content-type"
        ) || "";

    if (
        contentType.includes(
            "application/json"
        )
    ) {

        payload = await response.json();

    } else {

        payload = await response.text();

    }


    if (!response.ok) {

        let message =
            `API request failed (${response.status})`;

        if (
            payload &&
            typeof payload === "object"
        ) {

            if (payload.detail) {

                if (
                    typeof payload.detail ===
                    "string"
                ) {

                    message =
                        payload.detail;

                } else {

                    message =
                        JSON.stringify(
                            payload.detail
                        );
                }
            }
        }

        throw new Error(
            message
        );
    }

    return payload;
}


// ================================================================
// NAVIGATION
// ================================================================

function initializeNavigation() {

    const buttons =
        queryAll(
            ".nav-button"
        );

    buttons.forEach(
        button => {

            button.addEventListener(
                "click",
                () => {

                    const section =
                        button.dataset.section;

                    if (!section) {
                        return;
                    }

                    navigateToSection(
                        section
                    );
                }
            );
        }
    );


    queryAll(
        ".quick-action-card"
    ).forEach(
        button => {

            button.addEventListener(
                "click",
                async () => {

                    const section =
                        button.dataset.section;

                    if (section) {

                        navigateToSection(
                            section
                        );
                    }
                }
            );
        }
    );
}


function navigateToSection(
    sectionId
) {

    const sections =
        queryAll(
            ".application-section"
        );

    sections.forEach(
        section => {

            section.classList.toggle(
                "active-section",
                section.id === sectionId
            );
        }
    );


    queryAll(
        ".nav-button"
    ).forEach(
        button => {

            button.classList.toggle(
                "active",
                button.dataset.section ===
                sectionId
            );
        }
    );


    state.currentSection =
        sectionId;

    if (sectionId === "graph-section") {
        synchronizeSelectedTransaction();
    }

    window.scrollTo(
        {
            top: 0,
            behavior: "smooth"
        }
    );


    if (
        sectionId ===
        "system-section"
    ) {

        loadHealth();

    }


    if (
        sectionId ===
        "network-section"
    ) {

        loadNetworkOverview();

    }


    if (
        sectionId ===
        "alerts-section"
    ) {

        loadAlerts();

    }
}


// ================================================================
// OVERVIEW
// ================================================================

async function loadSummary() {

    try {

        const data =
            await apiFetch(
                "/summary"
            );

        state.summary =
            data;

        renderSummary(
            data
        );

    } catch (error) {

        console.error(
            "Summary error:",
            error
        );

        showSummaryError(
            error.message
        );
    }
}


function renderSummary(
    data
) {

    const transactions =
        data.transactions || {};

    const risk =
        data.risk || {};

    const alerts =
        data.alerts || {};

    const evidence =
        data.evidence || {};

    const topQueue =
        data.top_queue || {};


    setText(
        "metric-transactions",
        formatNumber(
            transactions.unique_txids
        )
    );

    setText(
        "metric-active",
        formatNumber(
            alerts.active
        )
    );


    const distribution =
        alerts.priority_distribution || {};


    setText(
        "metric-very-high",
        formatNumber(
            distribution.VERY_HIGH || 0
        )
    );


    setText(
        "metric-mean-risk",
        formatRisk(
            risk.mean
        )
    );


    setText(
        "metric-top-queue",
        formatNumber(
            topQueue.size
        )
    );


    setText(
        "metric-channels",
        Number.isFinite(
            Number(
                evidence.mean_evidence_channels
            )
        )
            ? Number(
                evidence.mean_evidence_channels
            ).toFixed(2)
            : "—"
    );
}


function showSummaryError(
    message
) {

    const cards =
        queryAll(
            ".metric-value"
        );

    cards.forEach(
        element => {
            element.textContent =
                "ERR";
        }
    );

    console.error(
        message
    );
}


// ================================================================
// TEMPORAL INTELLIGENCE
// ================================================================

function initializeTimeControls() {

    const start =
        $("time-start");

    const end =
        $("time-end");


    if (start) {

        start.innerHTML =
            "";

        for (
            let step = 1;
            step <= 49;
            step++
        ) {

            const option =
                document.createElement(
                    "option"
                );

            option.value =
                String(step);

            option.textContent =
                String(step);

            start.appendChild(
                option
            );
        }

        start.value = "1";
    }


    if (end) {

        end.innerHTML =
            "";

        for (
            let step = 1;
            step <= 49;
            step++
        ) {

            const option =
                document.createElement(
                    "option"
                );

            option.value =
                String(step);

            option.textContent =
                String(step);

            end.appendChild(
                option
            );
        }

        end.value = "49";
    }


    const refresh =
        $("refresh-temporal");

    if (refresh) {

        refresh.addEventListener(
            "click",
            loadTemporal
        );
    }
}


async function loadTemporal() {

    const start =
        Number(
            $("time-start")?.value || 1
        );

    const end =
        Number(
            $("time-end")?.value || 49
        );


    if (start > end) {

        setMessage(
            "txid-search-message",
            "Start time step cannot exceed end time step.",
            "error"
        );

        return;
    }


    setLoading(
        "temporal-chart",
        "Loading temporal intelligence..."
    );


    try {

        const data =
            await apiFetch(
                `/temporal?start_time_step=${start}&end_time_step=${end}`
            );

        state.temporal =
            data;

        renderTemporal(
            data
        );

    } catch (error) {

        console.error(
            "Temporal error:",
            error
        );

        setHtml(
            "temporal-chart",
            `
            <div class="error-panel">
                Unable to load temporal intelligence.
                <br>
                ${escapeHtml(error.message)}
            </div>
            `
        );
    }
}


function renderTemporal(
    data
) {

    const container =
        $("temporal-chart");

    if (!container) {
        return;
    }


    const rows =
        data.time_steps || [];


    if (!rows.length) {

        container.innerHTML =
            `
            <div class="empty-panel compact">
                No temporal data available.
            </div>
            `;

        return;
    }


    const maxRisk =
        Math.max(
            ...rows.map(
                row =>
                    Number(
                        row.mean_risk
                    ) || 0
            ),
            1
        );


    const maxTransactions =
        Math.max(
            ...rows.map(
                row =>
                    Number(
                        row.transaction_count
                    ) || 0
            ),
            1
        );


    const chartHeight = 220;


    let html =
        `
        <div class="temporal-chart-header">
            <div>
                <span class="chart-label">
                    Mean unified risk
                </span>
            </div>

            <div>
                <span class="chart-label">
                    Time steps ${data.start_time_step}–${data.end_time_step}
                </span>
            </div>
        </div>

        <div
            class="temporal-bars"
            style="height:${chartHeight}px"
        >
        `;


    rows.forEach(
        row => {

            const risk =
                Number(
                    row.mean_risk
                ) || 0;

            const transactions =
                Number(
                    row.transaction_count
                ) || 0;


            const riskHeight =
                Math.max(
                    2,
                    (
                        risk /
                        maxRisk
                    ) *
                    100
                );


            const transactionHeight =
                Math.max(
                    2,
                    (
                        transactions /
                        maxTransactions
                    ) *
                    100
                );


            const high =
                Number(
                    row.high_alert_count
                ) || 0;

            const veryHigh =
                Number(
                    row.very_high_alert_count
                ) || 0;


            html +=
                `
                <div
                    class="temporal-column"
                    title="Time ${row.time_step}: mean risk ${risk.toFixed(2)}, transactions ${formatNumber(transactions)}, high alerts ${formatNumber(high)}, very high ${formatNumber(veryHigh)}"
                >

                    <div class="temporal-bar-area">

                        <div
                            class="temporal-risk-bar"
                            style="height:${riskHeight}%"
                        ></div>

                    </div>

                    <div
                        class="temporal-transaction-indicator"
                        style="height:${transactionHeight}%"
                    ></div>

                    <div class="temporal-axis-label">
                        ${row.time_step}
                    </div>

                </div>
                `;
        }
    );


    html +=
        `
        </div>

        <div class="temporal-legend">

            <span>
                Mean risk
            </span>

            <span>
                Transaction volume
            </span>

        </div>
        `;


    container.innerHTML =
        html;
}


// ================================================================
// ALERTS
// ================================================================

function initializeAlertFilters() {

    const timeFilter =
        $("time-filter");


    if (timeFilter) {

        timeFilter.innerHTML =
            `<option value="">All time steps</option>`;

        for (
            let step = 1;
            step <= 49;
            step++
        ) {

            const option =
                document.createElement(
                    "option"
                );

            option.value =
                String(step);

            option.textContent =
                String(step);

            timeFilter.appendChild(
                option
            );
        }
    }


    $("apply-filters")
        ?.addEventListener(
            "click",
            () => {

                state.alertFilters = {

                    priority:
                        $("priority-filter")
                            ?.value || "",

                    status:
                        $("status-filter")
                            ?.value || "",

                    timeStep:
                        $("time-filter")
                            ?.value || "",

                    minRisk:
                        $("min-risk")
                            ?.value || "",

                    maxRisk:
                        $("max-risk")
                            ?.value || ""
                };

                state.currentPage =
                    1;

                loadAlerts();
            }
        );


    $("previous-page")
        ?.addEventListener(
            "click",
            () => {

                if (
                    state.currentPage > 1
                ) {

                    state.currentPage -=
                        1;

                    loadAlerts();
                }
            }
        );


    $("next-page")
        ?.addEventListener(
            "click",
            () => {

                if (
                    state.alerts &&
                    state.currentPage <
                    state.alerts.pages
                ) {

                    state.currentPage +=
                        1;

                    loadAlerts();
                }
            }
        );
}


function buildAlertQuery() {

    const filters =
        state.alertFilters;

    const params =
        new URLSearchParams();


    params.set(
        "page",
        String(
            state.currentPage
        )
    );

    params.set(
        "page_size",
        String(
            state.pageSize
        )
    );


    if (filters.priority) {

        params.set(
            "priority",
            filters.priority
        );
    }


    if (filters.status) {

        params.set(
            "alert_status",
            filters.status
        );
    }


    if (filters.timeStep) {

        params.set(
            "time_step",
            filters.timeStep
        );
    }


    if (filters.minRisk !== "") {

        params.set(
            "min_risk",
            filters.minRisk
        );
    }


    if (filters.maxRisk !== "") {

        params.set(
            "max_risk",
            filters.maxRisk
        );
    }


    return params.toString();
}


async function loadAlerts() {

    setLoading(
        "alerts-body",
        "Loading alerts..."
    );


    try {

        const query =
            buildAlertQuery();

        const data =
            await apiFetch(
                `/alerts?${query}`
            );

        state.alerts =
            data;

        renderAlerts(
            data
        );

    } catch (error) {

        console.error(
            "Alerts error:",
            error
        );

        setHtml(
            "alerts-body",
            `
            <tr>
                <td
                    colspan="12"
                    class="error-cell"
                >
                    Unable to load alerts:
                    ${escapeHtml(error.message)}
                </td>
            </tr>
            `
        );
    }
}


function renderAlerts(
    data
) {

    setText(
        "alert-total",
        `${formatNumber(data.total)} records`
    );


    const tbody =
        $("alerts-body");

    if (!tbody) {
        return;
    }


    const alerts =
        data.alerts || [];


    if (!alerts.length) {

        tbody.innerHTML =
            `
            <tr>
                <td
                    colspan="12"
                    class="empty-cell"
                >
                    No alerts match the current filters.
                </td>
            </tr>
            `;

        updatePagination(
            data
        );

        return;
    }


    tbody.innerHTML =
        alerts.map(
            alert => {

                const txid =
                    normalizeTxid(
                        alert.txid
                    );

                const risk =
                    Number(
                        alert.risk_score
                    );


                return `
                <tr
                    class="alert-row"
                    data-txid="${escapeHtml(txid)}"
                >

                    <td>
                        <span class="rank-number">
                            ${escapeHtml(
                                formatNumber(
                                    alert.investigation_rank
                                )
                            )}
                        </span>
                    </td>


                    <td>

                        <button
                            class="txid-button"
                            data-action="investigate"
                            data-txid="${escapeHtml(txid)}"
                            title="Investigate transaction"
                        >
                            ${escapeHtml(txid)}
                        </button>

                    </td>


                    <td>
                        ${escapeHtml(
                            alert.time_step
                        )}
                    </td>


                    <td>

                        <span
                            class="risk-pill ${riskClass(risk)}"
                        >
                            ${escapeHtml(
                                formatRisk(risk)
                            )}
                        </span>

                    </td>


                    <td>

                        <span
                            class="priority-pill ${priorityClass(
                                alert.alert_priority
                            )}"
                        >
                            ${escapeHtml(
                                alert.alert_priority
                            )}
                        </span>

                    </td>


                    <td>
                        ${escapeHtml(
                            formatPercent(
                                alert.ml_confidence,
                                1
                            )
                        )}
                    </td>


                    <td>
                        ${escapeHtml(
                            formatPercent(
                                alert.anomaly_signal,
                                1
                            )
                        )}
                    </td>


                    <td>
                        ${escapeHtml(
                            formatPercent(
                                alert.behavioral_signal,
                                1
                            )
                        )}
                    </td>


                    <td>
                        ${escapeHtml(
                            formatPercent(
                                alert.entity_signal,
                                1
                            )
                        )}
                    </td>


                    <td>
                        ${escapeHtml(
                            formatPercent(
                                alert.network_signal,
                                1
                            )
                        )}
                    </td>


                    <td>
                        ${escapeHtml(
                            alert.effective_evidence_channel_count
                        )}
                    </td>


                    <td>

                        <button
                            class="small-button"
                            data-action="investigate"
                            data-txid="${escapeHtml(txid)}"
                        >
                            Open
                        </button>

                    </td>

                </tr>
                `;
            }
        )
        .join("");


    queryAll(
        "[data-action='investigate']"
    ).forEach(
        button => {

            button.addEventListener(
                "click",
                () => {

                    const txid =
                        normalizeTxid(
                            button.dataset.txid
                        );

                    if (txid) {

                        openInvestigation(
                            txid
                        );
                    }
                }
            );
        }
    );


    updatePagination(
        data
    );
}


function updatePagination(
    data
) {

    setText(
        "page-information",
        data.pages
            ? `Page ${formatNumber(
                data.page
            )} of ${formatNumber(
                data.pages
            )}`
            : "No pages"
    );


    const previous =
        $("previous-page");

    const next =
        $("next-page");


    if (previous) {

        previous.disabled =
            data.page <= 1;
    }


    if (next) {

        next.disabled =
            data.page >= data.pages;
    }
}


// ================================================================
// INVESTIGATION
// ================================================================

function initializeInvestigation() {

    $("search-txid")
        ?.addEventListener(
            "click",
            () => {

                const txid =
                    normalizeTxid(
                        $("txid-search")
                            ?.value
                    );

                if (!txid) {

                    setMessage(
                        "txid-search-message",
                        "Enter a transaction ID.",
                        "warning"
                    );

                    return;
                }

                openInvestigation(
                    txid
                );
            }
        );


    $("txid-search")
        ?.addEventListener(
            "keydown",
            event => {

                if (
                    event.key ===
                    "Enter"
                ) {

                    event.preventDefault();

                    $("search-txid")
                        ?.click();
                }
            }
        );


    $("clear-investigation")
        ?.addEventListener(
            "click",
            clearInvestigation
        );


    $("generate-selected-report")
        ?.addEventListener(
            "click",
            () => {

                if (
                    state.selectedTxid
                ) {

                    openReportForTxid(
                        state.selectedTxid
                    );

                } else {

                    setMessage(
                        "txid-search-message",
                        "No transaction is currently selected.",
                        "warning"
                    );
                }
            }
        );
}


async function openInvestigation(
    txid
) {

    const normalized =
        normalizeTxid(txid);

    if (!normalized) {
        return;
    }


    state.selectedTxid =
        normalized;

    state.loadingInvestigation =
        true;


    $("txid-search").value =
        normalized;

    setText(
        "selected-txid",
        normalized
    );


    navigateToSection(
        "investigate-section"
    );


    hide(
        "investigation-empty"
    );

    show(
        "investigation-panel"
    );


    setLoading(
        "investigation-risk",
        "Loading..."
    );


    setLoading(
        "evidence-bars",
        "Loading evidence..."
    );


    setLoading(
        "model-details",
        "Loading..."
    );


    setLoading(
        "behavior-details",
        "Loading..."
    );


    setLoading(
        "transaction-details",
        "Loading..."
    );


    setLoading(
        "network-details",
        "Loading..."
    );


    setLoading(
        "shap-details",
        "Loading..."
    );


    setMessage(
        "txid-search-message",
        "Loading investigation profile..."
    );


    try {

        const data =
            await apiFetch(
                `/alerts/${encodeURIComponent(
                    normalized
                )}`
            );

        state.selectedInvestigation =
            data;

        renderInvestigation(
            data
        );

        setMessage(
            "txid-search-message",
            "Investigation profile loaded.",
            "success"
        );

    } catch (error) {

        console.error(
            "Investigation error:",
            error
        );

        state.selectedInvestigation =
            null;

        show(
            "investigation-empty"
        );

        hide(
            "investigation-panel"
        );

        setMessage(
            "txid-search-message",
            error.message,
            "error"
        );

    } finally {

        state.loadingInvestigation =
            false;
    }
}


function renderInvestigation(
    data
) {

    const alert =
        data.alert || {};

    const risk =
        data.risk || {};

    const temporal =
        (
            data.temporal_entity_evidence ||
            []
        );

    const shap =
        data.shap || [];


    renderInvestigationRisk(
        alert,
        risk
    );

    renderEvidence(
        alert
    );

    renderModelDetails(
        alert,
        risk
    );

    renderBehaviorDetails(
        alert,
        temporal
    );

    renderTransactionDetails(
        alert,
        risk,
        temporal
    );

    renderNetworkDetails(
        alert,
        risk
    );

    renderShap(
        shap
    );


    const reportTxid =
        $("report-txid");

    if (reportTxid) {

        reportTxid.value =
            normalizeTxid(
                data.txid
            );
    }
}


function renderInvestigationRisk(
    alert,
    risk
) {

    const score =
        Number(
            alert.risk_score ??
            risk.risk_score
        );


    const level =
        alert.risk_level ??
        risk.risk_level ??
        "—";


    const priority =
        alert.alert_priority ??
        "—";


    const explanation =
        alert.alert_explanation ??
        alert.risk_explanation ??
        risk.risk_explanation ??
        "No explanation available.";


    setHtml(
        "investigation-risk",
        `
        <div class="risk-score-large ${riskClass(score)}">
            ${escapeHtml(
                formatRisk(score)
            )}
        </div>

        <div class="risk-level">
            ${escapeHtml(level)}
        </div>

        <div class="risk-priority">
            ${escapeHtml(priority)}
        </div>
        `
    );


    setHtml(
        "investigation-explanation",
        `
        <div class="explanation-label">
            Investigative explanation
        </div>

        <p>
            ${escapeHtml(
                explanation
            )}
        </p>
        `
    );
}


function renderEvidence(
    alert
) {

    const channels = [

        {
            name: "ML",
            value:
                Number(
                    alert.ml_confidence
                ) || 0
        },

        {
            name: "Anomaly",
            value:
                Number(
                    alert.anomaly_signal
                ) || 0
        },

        {
            name: "Behavior",
            value:
                Number(
                    alert.behavioral_signal
                ) || 0
        },

        {
            name: "Entity",
            value:
                Number(
                    alert.entity_signal
                ) || 0
        },

        {
            name: "Network",
            value:
                Number(
                    alert.network_signal
                ) || 0
        }

    ];


    setHtml(
        "evidence-bars",
        channels.map(
            channel => {

                const percentage =
                    Math.max(
                        0,
                        Math.min(
                            100,
                            channel.value * 100
                        )
                    );


                return `
                <div class="evidence-row">

                    <div class="evidence-header">

                        <span>
                            ${escapeHtml(
                                channel.name
                            )}
                        </span>

                        <span>
                            ${percentage.toFixed(1)}%
                        </span>

                    </div>

                    <div class="evidence-track">

                        <div
                            class="evidence-fill"
                            style="width:${percentage}%"
                        ></div>

                    </div>

                </div>
                `;
            }
        ).join("")
    );
}


function renderModelDetails(
    alert,
    risk
) {

    const items = [

        [
            "Predicted class",
            alert.ml_predicted_class ??
            risk.ml_predicted_class
        ],

        [
            "ML confidence",
            formatPercent(
                alert.ml_confidence ??
                risk.ml_confidence,
                2
            )
        ],

        [
            "Investigation rank",
            formatNumber(
                alert.investigation_rank ??
                risk.investigation_rank
            )
        ],

        [
            "Alert status",
            alert.alert_status
        ],

        [
            "Evidence channels",
            alert.effective_evidence_channel_count
        ],

        [
            "Evidence agreement",
            formatPercent(
                alert.evidence_agreement,
                1
            )
        ]

    ];


    renderDetailGrid(
        "model-details",
        items
    );
}


function renderBehaviorDetails(
    alert,
    temporal
) {

    const items = [

        [
            "Behavioral score",
            formatPercent(
                alert.behavioral_signal,
                2
            )
        ],

        [
            "Peeling evidence",
            alert.peeling_signal ??
            alert.peeling_chain_signal ??
            "Not available"
        ],

        [
            "Mixing evidence",
            alert.mixing_signal ??
            "Not available"
        ],

        [
            "Behavior level",
            alert.behavior_level ??
            "Not available"
        ],

        [
            "Temporal evidence rows",
            temporal.length
        ]

    ];


    renderDetailGrid(
        "behavior-details",
        items
    );
}


function renderTransactionDetails(
    alert,
    risk,
    temporal
) {

    const items = [

        [
            "TXID",
            normalizeTxid(
                alert.txid ??
                risk.txid
            )
        ],

        [
            "Time step",
            alert.time_step ??
            risk.time_step
        ],

        [
            "Risk score",
            formatRisk(
                alert.risk_score ??
                risk.risk_score
            )
        ],

        [
            "Risk level",
            alert.risk_level ??
            risk.risk_level
        ],

        [
            "Alert priority",
            alert.alert_priority
        ],

        [
            "Top queue",
            alert.top_alert_queue
                ? "Yes"
                : "No"
        ],

        [
            "Temporal evidence",
            temporal.length
        ]

    ];


    renderDetailList(
        "transaction-details",
        items
    );
}


function renderNetworkDetails(
    alert,
    risk
) {

    const network =
        [
            [
                "Network signal",
                formatPercent(
                    alert.network_signal ??
                    risk.network_signal,
                    2
                )
            ],

            [
                "Network evidence",
                (
                    Number(
                        alert.network_signal ??
                        risk.network_signal
                    ) > 0
                )
                    ? "Present"
                    : "No correlated signal"
            ],

            [
                "VPN / Proxy / Tor",
                "See Network workspace"
            ],

            [
                "Network source",
                "Offline intelligence layer"
            ]

        ];


    renderDetailList(
        "network-details",
        network
    );
}


function renderShap(
    shap
) {

    const container =
        $("shap-details");

    if (!container) {
        return;
    }


    if (!shap.length) {

        container.innerHTML =
            `
            <div class="empty-panel compact">
                No SHAP explanation is available
                for this transaction in the current
                ranked explanation artifact.
            </div>
            `;

        return;
    }


    const record =
        shap[0];


    const excludedKeys =
        new Set(
            [
                "txid",
                "txid_normalized",
                "predicted_class",
                "prediction_confidence",
                "base_value"
            ]
        );


    const candidates =
        Object.entries(
            record
        )
        .filter(
            ([key, value]) =>
                !excludedKeys.has(key) &&
                Number.isFinite(
                    Number(value)
                )
        );


    candidates.sort(
        (
            first,
            second
        ) =>
            Math.abs(
                Number(second[1])
            )
            -
            Math.abs(
                Number(first[1])
            )
    );


    const top =
        candidates.slice(
            0,
            12
        );


    let html =
        `
        <div class="shap-summary">

            <div>
                Predicted class:
                <strong>
                    ${escapeHtml(
                        record.predicted_class ??
                        "—"
                    )}
                </strong>
            </div>

            <div>
                Confidence:
                <strong>
                    ${escapeHtml(
                        formatPercent(
                            record.prediction_confidence,
                            2
                        )
                    )}
                </strong>
            </div>

        </div>
        `;


    if (!top.length) {

        html +=
            `
            <div class="empty-panel compact">
                SHAP record loaded, but no numeric
                feature contributions were found.
            </div>
            `;

        container.innerHTML =
            html;

        return;
    }


    html +=
        `<div class="shap-items">`;


    top.forEach(
        ([key, value]) => {

            const numeric =
                Number(value);

            const sign =
                numeric >= 0
                    ? "positive"
                    : "negative";


            html +=
                `
                <div class="shap-item">

                    <div class="shap-feature">
                        ${escapeHtml(key)}
                    </div>

                    <div
                        class="shap-value ${sign}"
                    >
                        ${numeric >= 0 ? "+" : ""}
                        ${numeric.toFixed(4)}
                    </div>

                </div>
                `;
        }
    );


    html +=
        `</div>`;


    html +=
        `
        <div class="shap-note">
            SHAP values describe model feature
            contributions. They are not independent
            evidence of identity, intent or guilt.
        </div>
        `;


    container.innerHTML =
        html;
}


function renderDetailGrid(
    id,
    items
) {

    const container =
        $(id);

    if (!container) {
        return;
    }


    container.innerHTML =
        items.map(
            item => {

                const label =
                    item[0];

                const value =
                    item[1];


                return `
                <div class="detail-item">

                    <div class="detail-label">
                        ${escapeHtml(label)}
                    </div>

                    <div class="detail-value">
                        ${escapeHtml(
                            value === undefined ||
                            value === null ||
                            value === ""
                                ? "—"
                                : value
                        )}
                    </div>

                </div>
                `;
            }
        ).join("");
}


function renderDetailList(
    id,
    items
) {

    const container =
        $(id);

    if (!container) {
        return;
    }


    container.innerHTML =
        items.map(
            item => {

                return `
                <div class="detail-list-row">

                    <div class="detail-label">
                        ${escapeHtml(
                            item[0]
                        )}
                    </div>

                    <div class="detail-value">
                        ${escapeHtml(
                            item[1] === undefined ||
                            item[1] === null ||
                            item[1] === ""
                                ? "—"
                                : item[1]
                        )}
                    </div>

                </div>
                `;
            }
        ).join("");
}


function clearInvestigation() {

    state.selectedTxid =
        null;

    state.selectedAlert =
        null;

    state.selectedInvestigation =
        null;


    const search =
        $("txid-search");

    if (search) {
        search.value = "";
    }


    setText(
        "selected-txid",
        "No transaction selected"
    );


    hide(
        "investigation-panel"
    );

    show(
        "investigation-empty"
    );


    setMessage(
        "txid-search-message",
        ""
    );
}


// ================================================================
// GRAPH
// ================================================================

function initializeGraph() {

    $("search-graph")
        ?.addEventListener(
            "click",
            () => {

                const input =
                    $("graph-txid-search");

                const txid =
                    normalizeTxid(
                        input?.value
                    );

                if (!txid) {

                    setMessage(
                        "graph-search-message",
                        "Enter a transaction ID.",
                        "warning"
                    );

                    return;
                }

                openGraph(
                    txid
                );
            }
        );


    $("graph-txid-search")
        ?.addEventListener(
            "keydown",
            event => {

                if (
                    event.key ===
                    "Enter"
                ) {

                    event.preventDefault();

                    $("search-graph")
                        ?.click();
                }
            }
        );
}


async function openGraph(
    txid
) {

    const normalized =
        normalizeTxid(txid);

    if (!normalized) {
        return;
    }


    /*
     * Keep the currently investigated transaction
     * synchronized across workspaces.
     *
     * This allows:
     *
     * Alerts -> Investigation -> Graph
     *
     * without requiring the analyst to copy/paste
     * the TXID again.
     */
    state.selectedTxid =
        normalized;

    synchronizeSelectedTransaction();


    state.loadingGraph =
        true;


    navigateToSection(
        "graph-section"
    );


    $("graph-txid-search").value =
        normalized;


    hide(
        "graph-empty"
    );

    show(
        "graph-panel"
    );


    setMessage(
        "graph-search-message",
        "Loading graph neighborhood..."
    );


    setLoading(
        "graph-summary",
        "Loading graph..."
    );


    setLoading(
        "graph-visualization",
        "Building relationship view..."
    );


    setLoading(
        "graph-nodes",
        "Loading connected nodes..."
    );


    try {

        const data =
            await apiFetch(
                `/graph/${encodeURIComponent(
                    normalized
                )}`
            );


        state.selectedGraph =
            data;

        renderGraph(
            data
        );


        setMessage(
            "graph-search-message",
            "Graph neighborhood loaded.",
            "success"
        );

    } catch (error) {

        console.error(
            "Graph error:",
            error
        );


        hide(
            "graph-panel"
        );

        show(
            "graph-empty"
        );


        setMessage(
            "graph-search-message",
            error.message,
            "error"
        );

    } finally {

        state.loadingGraph =
            false;
    }
}


function renderGraph(
    data
) {

    const transaction =
        data.transaction || {};


    const nodes =
        data.nodes || [];


    const edges =
        data.edges || [];


    renderGraphSummary(
        data
    );


    renderGraphVisualization(
        transaction,
        nodes,
        edges
    );


    renderGraphNodes(
        nodes
    );
}


function renderGraphSummary(
    data
) {

    const transaction =
        data.transaction || {};


    const risk =
        Number(
            transaction.risk_score
        );


    setHtml(
        "graph-summary",
        `
        <div class="graph-stat">

            <div class="graph-stat-label">
                Transaction
            </div>

            <div class="graph-stat-value mono">
                ${escapeHtml(
                    normalizeTxid(
                        data.txid
                    )
                )}
            </div>

        </div>

        <div class="graph-stat">

            <div class="graph-stat-label">
                Connected nodes
            </div>

            <div class="graph-stat-value">
                ${escapeHtml(
                    formatNumber(
                        data.connected_node_count
                    )
                )}
            </div>

        </div>

        <div class="graph-stat">

            <div class="graph-stat-label">
                Risk
            </div>

            <div class="graph-stat-value ${riskClass(risk)}">
                ${escapeHtml(
                    formatRisk(risk)
                )}
            </div>

        </div>
        `
    );
}


function renderGraphVisualization(
    transaction,
    nodes,
    edges
) {

    const container =
        $("graph-visualization");

    if (!container) {
        return;
    }


    const centerId =
        `tx:${normalizeTxid(
            transaction.txid ??
            state.selectedTxid
        )}`;


    const usableNodes =
        nodes
            .filter(
                node =>
                    node &&
                    node.node_id
            )
            .slice(
                0,
                MAX_GRAPH_NODES
            );


    if (!usableNodes.length) {

        container.innerHTML =
            `
            <div class="empty-panel compact">
                No connected graph nodes are available
                for this transaction.
            </div>
            `;

        return;
    }


    /*
     * Lightweight offline graph visualization.
     *
     * No D3, Cytoscape or external graph library is used.
     * The layout is deterministic and intended for
     * investigative neighborhood inspection.
     */


    const width = 900;

    const height = 480;

    const centerX =
        width / 2;

    const centerY =
        height / 2;


    const radius =
        Math.min(
            185,
            70 +
            usableNodes.length * 5
        );


    const positions =
        new Map();


    positions.set(
        centerId,
        {
            x: centerX,
            y: centerY
        }
    );


    usableNodes.forEach(
        (node, index) => {

            const angle =
                (
                    index /
                    Math.max(
                        usableNodes.length,
                        1
                    )
                )
                *
                Math.PI *
                2;

            positions.set(
                node.node_id,
                {
                    x:
                        centerX +
                        Math.cos(
                            angle
                        ) *
                        radius,

                    y:
                        centerY +
                        Math.sin(
                            angle
                        ) *
                        radius
                }
            );
        }
    );


    const visibleEdges =
        edges.filter(
            edge =>
                positions.has(
                    edge.source
                ) ||
                positions.has(
                    edge.target
                )
        );


    let svg =
        `
        <svg
            class="investigation-graph-svg"
            viewBox="0 0 ${width} ${height}"
            role="img"
            aria-label="Transaction investigation graph"
        >

        <rect
            x="0"
            y="0"
            width="${width}"
            height="${height}"
            class="graph-background"
        >
        </rect>
        `;


    visibleEdges.forEach(
        edge => {

            const source =
                positions.get(
                    edge.source
                );

            const target =
                positions.get(
                    edge.target
                );


            if (!source || !target) {
                return;
            }


            svg +=
                `
                <line
                    x1="${source.x}"
                    y1="${source.y}"
                    x2="${target.x}"
                    y2="${target.y}"
                    class="graph-edge"
                ></line>
                `;
        }
    );


    svg +=
        `
        <circle
            cx="${centerX}"
            cy="${centerY}"
            r="27"
            class="graph-center-node"
        ></circle>

        <text
            x="${centerX}"
            y="${centerY + 4}"
            text-anchor="middle"
            class="graph-center-label"
        >
            TX
        </text>
        `;


    usableNodes.forEach(
        (node, index) => {

            const position =
                positions.get(
                    node.node_id
                );


            if (!position) {
                return;
            }


            const type =
                String(
                    node.node_type ||
                    node.type ||
                    ""
                )
                .toUpperCase();


            let nodeClass =
                "graph-wallet-node";


            if (
                type.includes(
                    "TRANSACTION"
                ) ||
                String(
                    node.node_id
                ).startsWith(
                    "tx:"
                )
            ) {

                nodeClass =
                    "graph-transaction-node";

            } else if (
                type.includes(
                    "IP"
                ) ||
                String(
                    node.node_id
                ).startsWith(
                    "ip:"
                )
            ) {

                nodeClass =
                    "graph-ip-node";
            }


            svg +=
                `
                <circle
                    cx="${position.x}"
                    cy="${position.y}"
                    r="17"
                    class="${nodeClass}"
                ></circle>

                <text
                    x="${position.x}"
                    y="${position.y + 31}"
                    text-anchor="middle"
                    class="graph-node-label"
                >
                    ${escapeHtml(
                        graphNodeShortLabel(
                            node
                        )
                    )}
                </text>
                `;
        }
    );


    svg +=
        `
        </svg>

        <div class="graph-legend">

            <span>
                TX Transaction
            </span>

            <span>
                Wallet
            </span>

            <span>
                IP
            </span>

        </div>
        `;


    container.innerHTML =
        svg;
}


function graphNodeShortLabel(
    node
) {

    const nodeId =
        String(
            node.node_id ||
            ""
        );


    if (
        nodeId.startsWith(
            "tx:"
        )
    ) {

        return (
            nodeId
                .slice(3, 13)
            + "…"
        );
    }


    if (
        nodeId.startsWith(
            "wallet:"
        )
    ) {

        return (
            nodeId
                .slice(7, 17)
            + "…"
        );
    }


    if (
        nodeId.startsWith(
            "ip:"
        )
    ) {

        return (
            nodeId
                .slice(3, 18)
        );
    }


    if (
        nodeId.length > 15
    ) {

        return (
            nodeId.slice(
                0,
                15
            )
            + "…"
        );
    }


    return nodeId;
}


function renderGraphNodes(
    nodes
) {

    const container =
        $("graph-nodes");

    if (!container) {
        return;
    }


    if (!nodes.length) {

        container.innerHTML =
            `
            <div class="empty-panel compact">
                No connected nodes.
            </div>
            `;

        return;
    }


    container.innerHTML =
        nodes
            .slice(
                0,
                MAX_GRAPH_NODES
            )
            .map(
                node => {

                    const nodeId =
                        node.node_id ||
                        "—";

                    const type =
                        node.node_type ||
                        node.type ||
                        "UNKNOWN";


                    return `
                    <div class="graph-node-row">

                        <div>

                            <div class="graph-node-type">
                                ${escapeHtml(
                                    type
                                )}
                            </div>

                            <div class="graph-node-id mono">
                                ${escapeHtml(
                                    nodeId
                                )}
                            </div>

                        </div>

                        <button
                            class="small-button graph-investigate-button"
                            data-node-id="${escapeHtml(
                                nodeId
                            )}"
                        >
                            Inspect
                        </button>

                    </div>
                    `;
                }
            )
            .join("");


    queryAll(
        ".graph-investigate-button"
    ).forEach(
        button => {

            button.addEventListener(
                "click",
                async () => {

                    const nodeId =
                        button.dataset.nodeId ||
                        "";

                    if (
                        nodeId.startsWith(
                            "tx:"
                        )
                    ) {

                        openGraph(
                            nodeId.slice(
                                3
                            )
                        );

                    } else if (
                        nodeId.startsWith(
                            "wallet:"
                        )
                    ) {

                        if (
                            typeof window.inspectWalletFromGraph ===
                            "function"
                        ) {

                            window.inspectWalletFromGraph(
                                nodeId
                            );

                        } else {

                            setMessage(
                                "graph-search-message",
                                "Wallet investigation workspace is unavailable.",
                                "error"
                            );
                        }

                    } else {

                        setMessage(
                            "graph-search-message",
                            "Detailed inspection is available for transaction and wallet nodes.",
                            "warning"
                        );
                    }
                }
            );
        }
    );
}


// ================================================================
// NETWORK WORKSPACE
// ================================================================

async function loadNetworkOverview() {

    /*
     * M13 network intelligence is integrated into the
     * unified risk artifact. The API summary exposes
     * aggregate network evidence.
     *
     * Detailed VPN/proxy/Tor counts are read from the
     * local M13 fixture metadata when available through
     * the health/import API. We do not fabricate values.
     */

    try {

        const summary =
            state.summary ||
            await apiFetch(
                "/summary"
            );


        const networkSignal =
            Number(
                summary.evidence
                    ?.mean_network_signal
            );


        setText(
            "network-observation-count",
            Number.isFinite(
                networkSignal
            )
                ? (
                    networkSignal > 0
                        ? "Correlated"
                        : "0"
                )
                : "—"
        );


        /*
         * The current M13 fixture contains:
         * VPN = 1
         * Proxy = 1
         * Tor = 1
         * Hosting = 1
         *
         * These are displayed only if the local
         * fixture exists and can be established
         * from the health metadata.
         *
         * Otherwise the UI remains explicit.
         */

        let networkMetadata = null;

        try {

            const health =
                state.health ||
                await apiFetch(
                    "/health"
                );

            networkMetadata =
                health.network ||
                null;

        } catch (
            healthError
        ) {

            console.debug(
                "Network health metadata unavailable:",
                healthError
            );
        }


        if (
            networkMetadata &&
            typeof networkMetadata ===
            "object"
        ) {

            setText(
                "network-observation-count",
                formatNumber(
                    networkMetadata.correlated_transactions
                )
            );

            setText(
                "network-vpn-count",
                formatNumber(
                    networkMetadata.vpn
                )
            );

            setText(
                "network-proxy-count",
                formatNumber(
                    networkMetadata.proxy
                )
            );

            setText(
                "network-tor-count",
                formatNumber(
                    networkMetadata.tor
                )
            );

            setText(
                "network-hosting-count",
                formatNumber(
                    networkMetadata.hosting
                )
            );

        } else {

            setText(
                "network-vpn-count",
                "—"
            );

            setText(
                "network-proxy-count",
                "—"
            );

            setText(
                "network-tor-count",
                "—"
            );

            setText(
                "network-hosting-count",
                "—"
            );
        }

    } catch (error) {

        console.error(
            "Network overview error:",
            error
        );
    }
}


// ================================================================
// IMPORT
// ================================================================

function initializeImport() {

    const fileInput =
        $("import-file");


    fileInput?.addEventListener(
        "change",
        () => {

            const file =
                fileInput.files?.[0];


            if (!file) {

                setText(
                    "import-file-info",
                    "No file selected."
                );

                return;
            }


            const maxBytes =
                100 *
                1024 *
                1024;


            if (
                file.size >
                maxBytes
            ) {

                setText(
                    "import-file-info",
                    `File exceeds the 100 MB application limit: ${formatNumber(
                        file.size / 1024 / 1024,
                        2
                    )} MB`
                );

                setMessage(
                    "import-message",
                    "Select a file smaller than 100 MB.",
                    "error"
                );

                return;
            }


            setText(
                "import-file-info",
                `${file.name} — ${formatNumber(
                    file.size / 1024,
                    1
                )} KB`
            );

            setMessage(
                "import-message",
                ""
            );
        }
    );


    $("import-submit")
        ?.addEventListener(
            "click",
            submitImport
        );
}


async function submitImport() {

    if (state.importing) {
        return;
    }


    const fileInput =
        $("import-file");


    const dataType =
        $("import-data-type")
            ?.value;


    const file =
        fileInput?.files?.[0];


    if (!dataType) {

        setMessage(
            "import-message",
            "Select an investigator data type.",
            "warning"
        );

        return;
    }


    if (!file) {

        setMessage(
            "import-message",
            "Select a CSV, JSON or XML file.",
            "warning"
        );

        return;
    }


    const extension =
        file.name
            .split(".")
            .pop()
            ?.toLowerCase();


    if (
        ![
            "csv",
            "json",
            "xml"
        ].includes(
            extension
        )
    ) {

        setMessage(
            "import-message",
            "Only CSV, JSON and XML files are supported.",
            "error"
        );

        return;
    }


    state.importing =
        true;


    const submit =
        $("import-submit");


    if (submit) {

        submit.disabled =
            true;

        submit.textContent =
            "Processing...";
    }


    setMessage(
        "import-message",
        "Processing locally through the investigator import pipeline..."
    );


    setLoading(
        "import-result",
        "Validating and importing evidence..."
    );


    try {

        const formData =
            new FormData();


        formData.append(
            "file",
            file
        );


        const url =
            `${API_BASE}/import?data_type=${encodeURIComponent(
                dataType
            )}`;


        const response =
            await fetch(
                url,
                {
                    method: "POST",
                    body: formData
                }
            );


        let data = null;

        try {

            data =
                await response.json();

        } catch (
            parseError
        ) {

            data = {
                detail:
                    await response.text()
            };
        }


        if (!response.ok) {

            let message =
                `Import failed (${response.status})`;

            if (
                data &&
                data.detail
            ) {

                message =
                    typeof data.detail ===
                    "string"
                        ? data.detail
                        : JSON.stringify(
                            data.detail
                        );
            }

            throw new Error(
                message
            );
        }


        renderImportResult(
            data
        );


        setMessage(
            "import-message",
            "Import completed successfully.",
            "success"
        );

    } catch (error) {

        console.error(
            "Import error:",
            error
        );


        setHtml(
            "import-result",
            `
            <div class="error-panel">
                <strong>
                    Import failed
                </strong>

                <p>
                    ${escapeHtml(
                        error.message
                    )}
                </p>
            </div>
            `
        );


        setMessage(
            "import-message",
            error.message,
            "error"
        );

    } finally {

        state.importing =
            false;


        if (submit) {

            submit.disabled =
                false;

            submit.textContent =
                "Validate and Import";
        }
    }
}


function renderImportResult(
    data
) {

    const statistics =
        data.statistics || {};

    const source =
        data.source || {};

    const provenance =
        data.provenance || {};

    const artifacts =
        data.artifacts || {};


    const status =
        String(
            data.status ||
            "UNKNOWN"
        );


    setHtml(
        "import-result",
        `
        <div class="import-result-header">

            <span class="priority-pill priority-${escapeHtml(
                status.toLowerCase()
            )}">
                ${escapeHtml(status)}
            </span>

            <span class="mono">
                ${escapeHtml(
                    data.import_id ||
                    "—"
                )}
            </span>

        </div>


        <div class="import-stat-grid">

            <div class="import-stat">
                <span>
                    Records read
                </span>

                <strong>
                    ${escapeHtml(
                        formatNumber(
                            statistics.records_read
                        )
                    )}
                </strong>
            </div>


            <div class="import-stat">
                <span>
                    Valid records
                </span>

                <strong>
                    ${escapeHtml(
                        formatNumber(
                            statistics.valid_records
                        )
                    )}
                </strong>
            </div>


            <div class="import-stat">
                <span>
                    Invalid records
                </span>

                <strong>
                    ${escapeHtml(
                        formatNumber(
                            statistics.invalid_records
                        )
                    )}
                </strong>
            </div>


            <div class="import-stat">
                <span>
                    Duplicates removed
                </span>

                <strong>
                    ${escapeHtml(
                        formatNumber(
                            statistics.duplicates_removed
                        )
                    )}
                </strong>
            </div>

        </div>


        <div class="import-section-block">

            <div class="panel-title">
                Source
            </div>

            <div class="detail-list">

                <div class="detail-list-row">

                    <span>
                        Filename
                    </span>

                    <strong>
                        ${escapeHtml(
                            source.filename
                        )}
                    </strong>

                </div>


                <div class="detail-list-row">

                    <span>
                        Format
                    </span>

                    <strong>
                        ${escapeHtml(
                            source.format
                        )}
                    </strong>

                </div>


                <div class="detail-list-row">

                    <span>
                        SHA-256
                    </span>

                    <strong class="mono hash-value">
                        ${escapeHtml(
                            source.sha256
                        )}
                    </strong>

                </div>

            </div>

        </div>


        <div class="import-section-block">

            <div class="panel-title">
                Provenance
            </div>

            <div class="detail-list">

                <div class="detail-list-row">

                    <span>
                        Import ID
                    </span>

                    <strong class="mono">
                        ${escapeHtml(
                            provenance.import_id
                        )}
                    </strong>

                </div>


                <div class="detail-list-row">

                    <span>
                        Imported at
                    </span>

                    <strong>
                        ${escapeHtml(
                            provenance.imported_at_utc
                        )}
                    </strong>

                </div>


                <div class="detail-list-row">

                    <span>
                        Offline processing
                    </span>

                    <strong>
                        ${provenance.offline_processing
                            ? "YES"
                            : "NO"}
                    </strong>

                </div>

            </div>

        </div>


        <div class="import-section-block">

            <div class="panel-title">
                Generated Artifacts
            </div>

            <div class="detail-list">

                <div class="detail-list-row">

                    <span>
                        Directory
                    </span>

                    <strong class="mono">
                        ${escapeHtml(
                            artifacts.directory
                        )}
                    </strong>

                </div>


                <div class="detail-list-row">

                    <span>
                        Normalized records
                    </span>

                    <strong class="mono">
                        ${escapeHtml(
                            artifacts.normalized_records
                        )}
                    </strong>

                </div>


                <div class="detail-list-row">

                    <span>
                        Validation errors
                    </span>

                    <strong class="mono">
                        ${escapeHtml(
                            artifacts.validation_errors
                        )}
                    </strong>

                </div>


                <div class="detail-list-row">

                    <span>
                        Manifest
                    </span>

                    <strong class="mono">
                        ${escapeHtml(
                            artifacts.manifest
                        )}
                    </strong>

                </div>

            </div>

        </div>
        `
    );


    const invalid =
        data.invalid_record_preview ||
        [];


    if (invalid.length) {

        const result =
            $("import-result");

        if (result) {

            result.innerHTML +=
                `
                <div class="import-section-block">

                    <div class="panel-title">
                        Invalid Record Preview
                    </div>

                    <div class="import-invalid-preview">

                        ${invalid
                            .map(
                                record =>
                                    `
                                    <pre>${escapeHtml(
                                        JSON.stringify(
                                            record,
                                            null,
                                            2
                                        )
                                    )}</pre>
                                    `
                            )
                            .join("")}

                    </div>

                </div>
                `;
        }
    }
}


// ================================================================
// REPORTS
// ================================================================

function initializeReports() {

    $("generate-report")
        ?.addEventListener(
            "click",
            () => {

                const txid =
                    normalizeTxid(
                        $("report-txid")
                            ?.value
                    );

                if (!txid) {

                    setMessage(
                        "report-message",
                        "Enter a transaction ID.",
                        "warning"
                    );

                    return;
                }

                generateReport(
                    txid
                );
            }
        );
}


function openReportForTxid(
    txid
) {

    const normalized =
        normalizeTxid(txid);

    if (!normalized) {
        return;
    }


    const input =
        $("report-txid");

    if (input) {

        input.value =
            normalized;
    }


    navigateToSection(
        "reports-section"
    );


    setMessage(
        "report-message",
        `Transaction ${normalized} selected for reporting.`,
        "success"
    );
}


async function generateReport(
    txid
) {

    const normalized =
        normalizeTxid(txid);

    if (!normalized) {
        return;
    }


    setMessage(
        "report-message",
        "Collecting investigation evidence..."
    );


    setLoading(
        "report-result",
        "Generating local investigation report..."
    );


    try {

        const data =
            await apiFetch(
                `/alerts/${encodeURIComponent(
                    normalized
                )}`
            );


        const report =
            buildClientSideReport(
                data
            );


        renderGeneratedReport(
            report
        );


        setMessage(
            "report-message",
            "Investigation report generated locally.",
            "success"
        );

    } catch (error) {

        console.error(
            "Report error:",
            error
        );


        setHtml(
            "report-result",
            `
            <div class="error-panel">
                Unable to generate report.
                <br>
                ${escapeHtml(
                    error.message
                )}
            </div>
            `
        );


        setMessage(
            "report-message",
            error.message,
            "error"
        );
    }
}


function buildClientSideReport(
    data
) {

    const alert =
        data.alert || {};

    const risk =
        data.risk || {};

    const temporal =
        data.temporal_entity_evidence ||
        [];

    const shap =
        data.shap ||
        [];


    return {

        report_type:
            "Bitcoin Transaction Investigation Report",

        generated_at_utc:
            new Date().toISOString(),

        offline:
            true,

        txid:
            normalizeTxid(
                data.txid
            ),

        risk: {

            score:
                alert.risk_score ??
                risk.risk_score,

            level:
                alert.risk_level ??
                risk.risk_level,

            priority:
                alert.alert_priority,

            status:
                alert.alert_status,

            investigation_rank:
                alert.investigation_rank

        },

        model: {

            predicted_class:
                alert.ml_predicted_class,

            confidence:
                alert.ml_confidence

        },

        evidence: {

            anomaly:
                alert.anomaly_signal,

            behavioral:
                alert.behavioral_signal,

            entity:
                alert.entity_signal,

            network:
                alert.network_signal,

            evidence_channels:
                alert.effective_evidence_channel_count,

            agreement:
                alert.evidence_agreement

        },

        explanations: {

            alert:
                alert.alert_explanation,

            risk:
                alert.risk_explanation,

            evidence_summary:
                alert.evidence_summary

        },

        temporal_entity_evidence:
            temporal,

        shap:
            shap,

        limitation:
            "This report contains investigative prioritization and supporting evidence. It does not establish real-world identity, ownership, intent, illicit activity or guilt."

    };
}


function renderGeneratedReport(
    report
) {

    const risk =
        Number(
            report.risk.score
        );


    const shapCount =
        Array.isArray(
            report.shap
        )
            ? report.shap.length
            : 0;


    setHtml(
        "report-result",
        `
        <div class="generated-report">

            <div class="generated-report-header">

                <div>

                    <div class="section-kicker">
                        INVESTIGATION REPORT
                    </div>

                    <h3>
                        ${escapeHtml(
                            report.txid
                        )}
                    </h3>

                </div>


                <button
                    id="download-report"
                    class="button primary"
                >
                    Export JSON
                </button>

            </div>


            <div class="report-risk-summary">

                <div>

                    <span>
                        Unified risk
                    </span>

                    <strong class="${riskClass(risk)}">
                        ${escapeHtml(
                            formatRisk(risk)
                        )}
                    </strong>

                </div>


                <div>

                    <span>
                        Priority
                    </span>

                    <strong>
                        ${escapeHtml(
                            report.risk.priority
                        )}
                    </strong>

                </div>


                <div>

                    <span>
                        Investigation rank
                    </span>

                    <strong>
                        ${escapeHtml(
                            formatNumber(
                                report.risk.investigation_rank
                            )
                        )}
                    </strong>

                </div>

            </div>


            <div class="report-section">

                <div class="panel-title">
                    Model
                </div>

                <p>
                    Predicted class:
                    <strong>
                        ${escapeHtml(
                            report.model.predicted_class
                        )}
                    </strong>
                </p>

                <p>
                    Confidence:
                    <strong>
                        ${escapeHtml(
                            formatPercent(
                                report.model.confidence,
                                2
                            )
                        )}
                    </strong>
                </p>

            </div>


            <div class="report-section">

                <div class="panel-title">
                    Evidence
                </div>

                <div class="report-evidence-grid">

                    <div>
                        Anomaly
                        <strong>
                            ${escapeHtml(
                                formatPercent(
                                    report.evidence.anomaly,
                                    2
                                )
                            )}
                        </strong>
                    </div>

                    <div>
                        Behavioral
                        <strong>
                            ${escapeHtml(
                                formatPercent(
                                    report.evidence.behavioral,
                                    2
                                )
                            )}
                        </strong>
                    </div>

                    <div>
                        Entity
                        <strong>
                            ${escapeHtml(
                                formatPercent(
                                    report.evidence.entity,
                                    2
                                )
                            )}
                        </strong>
                    </div>

                    <div>
                        Network
                        <strong>
                            ${escapeHtml(
                                formatPercent(
                                    report.evidence.network,
                                    2
                                )
                            )}
                        </strong>
                    </div>

                </div>

            </div>


            <div class="report-section">

                <div class="panel-title">
                    Explanation
                </div>

                <p>
                    ${escapeHtml(
                        report.explanations.alert ||
                        report.explanations.risk ||
                        "No explanation available."
                    )}
                </p>

            </div>


            <div class="report-section">

                <div class="panel-title">
                    Explainability
                </div>

                <p>
                    SHAP records available:
                    <strong>
                        ${escapeHtml(
                            shapCount
                        )}
                    </strong>
                </p>

            </div>


            <div class="report-section report-limitation">

                <div class="panel-title">
                    Limitation
                </div>

                <p>
                    ${escapeHtml(
                        report.limitation
                    )}
                </p>

            </div>


            <div class="report-meta">

                Generated:
                ${escapeHtml(
                    report.generated_at_utc
                )}

                ·

                Offline:
                ${report.offline ? "YES" : "NO"}

            </div>

        </div>
        `
    );


    $("download-report")
        ?.addEventListener(
            "click",
            () => {

                downloadJson(
                    report,
                    `bitcoin_investigation_${report.txid}.json`
                );
            }
        );
}


function downloadJson(
    data,
    filename
) {

    const json =
        JSON.stringify(
            data,
            null,
            2
        );


    const blob =
        new Blob(
            [
                json
            ],
            {
                type:
                    "application/json"
            }
        );


    const url =
        URL.createObjectURL(
            blob
        );


    const anchor =
        document.createElement(
            "a"
        );


    anchor.href =
        url;

    anchor.download =
        filename;

    document.body.appendChild(
        anchor
    );

    anchor.click();

    anchor.remove();


    setTimeout(
        () => {
            URL.revokeObjectURL(
                url
            );
        },
        1000
    );
}


// ================================================================
// SYSTEM HEALTH
// ================================================================

function initializeHealth() {

    $("refresh-health")
        ?.addEventListener(
            "click",
            loadHealth
        );
}


async function loadHealth() {

    const statusText =
        $("status-text");

    const indicator =
        $("status-indicator");


    if (statusText) {

        statusText.textContent =
            "Checking...";
    }


    if (indicator) {

        indicator.classList.remove(
            "healthy",
            "degraded",
            "error"
        );
    }


    try {

        const data =
            await apiFetch(
                "/health"
            );


        state.health =
            data;


        renderHealth(
            data
        );


        if (statusText) {

            statusText.textContent =
                String(
                    data.status ||
                    "UNKNOWN"
                ).toUpperCase();
        }


        if (indicator) {

            indicator.classList.add(
                data.status ===
                "healthy"
                    ? "healthy"
                    : "degraded"
            );
        }


    } catch (error) {

        console.error(
            "Health error:",
            error
        );


        if (statusText) {

            statusText.textContent =
                "ERROR";
        }


        if (indicator) {

            indicator.classList.add(
                "error"
            );
        }


        setHealthStatus(
            "health-api",
            "ERROR"
        );
    }
}


function renderHealth(
    data
) {

    const artifacts =
        data.artifacts || {};


    setHealthStatus(
        "health-api",
        "ONLINE"
    );


    setHealthStatus(
        "health-ranked-alerts",
        artifacts.ranked_alerts
    );


    setHealthStatus(
        "health-risk-scores",
        artifacts.risk_scores
    );


    setHealthStatus(
        "health-temporal",
        artifacts.temporal_graph_features
    );


    setHealthStatus(
        "health-shap",
        artifacts.shap_explanations
    );


    setHealthStatus(
        "health-graph",
        (
            artifacts.investigation_graph_nodes &&
            artifacts.investigation_graph_edges
        )
    );


    setHealthStatus(
        "health-dashboard",
        artifacts.dashboard
    );


    setHealthStatus(
        "health-import",
        artifacts.investigator_import_engine
    );


    setText(
        "system-version",
        data.version
    );


    const formats =
        data.import
            ?.supported_formats;


    setText(
        "system-formats",
        Array.isArray(
            formats
        )
            ? formats.join(
                ", "
            )
            : "—"
    );


    const size =
        Number(
            data.import
                ?.max_file_size_mb
        );


    setText(
        "system-import-size",
        Number.isFinite(size)
            ? `${size.toFixed(0)} MB`
            : "—"
    );
}


function setHealthStatus(
    id,
    value
) {

    const element =
        $(id);

    if (!element) {
        return;
    }


    let healthy = false;

    let label = "UNKNOWN";


    if (
        value === true
    ) {

        healthy = true;
        label = "AVAILABLE";

    } else if (
        value === false
    ) {

        healthy = false;
        label = "MISSING";

    } else if (
        typeof value ===
        "string"
    ) {

        healthy =
            value.toLowerCase()
                .includes(
                    "online"
                )
            ||
            value.toLowerCase()
                .includes(
                    "available"
                );

        label =
            value.toUpperCase();

    } else {

        label =
            "UNKNOWN";
    }


    element.textContent =
        label;


    element.classList.toggle(
        "health-good",
        healthy
    );

    element.classList.toggle(
        "health-bad",
        !healthy
    );
}


// ================================================================
// CROSS-WORKSPACE ACTIONS
// ================================================================

function synchronizeSelectedTransaction() {

    if (
        !state.selectedTxid
    ) {
        return;
    }


    const txid =
        state.selectedTxid;


    const graphInput =
        $("graph-txid-search");

    if (graphInput) {

        graphInput.value =
            txid;
    }


    const reportInput =
        $("report-txid");

    if (reportInput) {

        reportInput.value =
            txid;
    }
}


// ================================================================
// KEYBOARD SHORTCUTS
// ================================================================

function initializeKeyboardShortcuts() {

    document.addEventListener(
        "keydown",
        event => {

            if (
                event.ctrlKey &&
                event.key === "k"
            ) {

                event.preventDefault();

                navigateToSection(
                    "investigate-section"
                );

                $("txid-search")
                    ?.focus();

                return;
            }


            if (
                event.key ===
                "Escape"
            ) {

                if (
                    state.currentSection ===
                    "investigate-section"
                ) {

                    clearInvestigation();
                }
            }
        }
    );
}


// ================================================================
// INITIALIZATION
// ================================================================

async function initializeApplication() {

    console.log(
        "Bitcoin Transaction Intelligence Platform starting..."
    );


    initializeNavigation();

    initializeTimeControls();

    initializeAlertFilters();

    initializeInvestigation();

    initializeGraph();

    initializeImport();

    initializeReports();

    initializeHealth();

    initializeKeyboardShortcuts();


    synchronizeSelectedTransaction();


    /*
     * Load the most important information in parallel.
     * A failure in one workspace does not prevent the
     * others from loading.
     */

    await Promise.allSettled(
        [
            loadHealth(),
            loadSummary(),
            loadTemporal(),
            loadAlerts()
        ]
    );


    /*
     * Network overview depends on health/summary
     * where available.
     */

    await loadNetworkOverview();


    console.log(
        "Bitcoin Transaction Intelligence Platform ready."
    );
}


document.addEventListener(
    "DOMContentLoaded",
    initializeApplication
);
