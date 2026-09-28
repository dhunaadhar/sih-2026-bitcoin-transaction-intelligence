(() => {
    "use strict";

    const state = { importId: "", txid: "", messages: [] };
    const $ = (id) => document.getElementById(id);

    function addMessage(role, text) {
        const container = $("assistant-conversation");
        if (!container) return;

        const article = document.createElement("article");
        article.className = `assistant-message ${role}`;

        const label = document.createElement("div");
        label.className = "assistant-message-label";
        label.textContent = role === "user"
            ? "INVESTIGATOR"
            : "INVESTIGATION ASSISTANT";

        const body = document.createElement("div");
        body.className = "assistant-message-body";
        body.textContent = text;

        article.append(label, body);
        container.appendChild(article);
        container.scrollTop = container.scrollHeight;
        state.messages.push({ role, text });
    }

    function setStatus(text, mode = "ready") {
        const status = $("assistant-status");
        if (status) {
            status.textContent = text;
            status.dataset.mode = mode;
        }
    }

    function setContext() {
        state.importId = $("assistant-import-id")?.value.trim() || "";
        state.txid = $("assistant-txid")?.value.trim() || "";

        if ($("assistant-context-import")) {
            $("assistant-context-import").textContent =
                state.importId || "No case selected";
        }
        if ($("assistant-context-txid")) {
            $("assistant-context-txid").textContent =
                state.txid || "Case level";
        }
    }

    async function askAssistant(question) {
        const text = question.trim();
        if (!text) return;

        setContext();

        if (!state.importId) {
            setStatus("Case ID required", "error");
            addMessage(
                "assistant",
                "Please provide an imported case ID before starting the investigation."
            );
            return;
        }

        addMessage("user", text);
        setStatus("Analyzing imported evidence...", "working");

        try {
            const response = await fetch("/api/assistant/query", {
                method: "POST",
                credentials: "same-origin",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: text,
                    import_id: state.importId,
                    txid: state.txid || null
                })
            });

            const payload = await response.json();

            if (!response.ok) {
                throw new Error(
                    payload?.detail ||
                    "The investigation request could not be completed."
                );
            }

            addMessage(
                "assistant",
                payload.answer || "No response was returned."
            );
            setStatus("Ready", "ready");
        } catch (error) {
            addMessage(
                "assistant",
                `Investigation request could not be completed. ${error.message}`
            );
            setStatus("Request failed", "error");
        }
    }

    function submitQuestion() {
        const input = $("assistant-question");
        if (!input) return;

        const question = input.value.trim();
        if (!question) return;

        input.value = "";
        askAssistant(question);
    }

    function initializeAssistant() {
        const send = $("assistant-send");
        const input = $("assistant-question");

        if (!send || !input) return;

        $("assistant-import-id")?.addEventListener("input", setContext);
        $("assistant-txid")?.addEventListener("input", setContext);

        send.addEventListener("click", submitQuestion);

        input.addEventListener("keydown", (event) => {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submitQuestion();
            }
        });

        document.querySelectorAll(".assistant-suggestion").forEach((button) => {
            button.addEventListener("click", () => {
                input.value = button.dataset.question || "";
                input.focus();
            });
        });

        setContext();

        addMessage(
            "assistant",
            "Good morning. I’m the Investigation Assistant. I’ll keep the analysis evidence-based and limited to the imported case data. Enter an imported case ID to begin."
        );
    }

    window.initializeInvestigationAssistant = initializeAssistant;
    window.askInvestigationAssistant = askAssistant;

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initializeAssistant);
    } else {
        initializeAssistant();
    }
})();
