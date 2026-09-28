(() => {
    "use strict";

    const AUTH_LOGIN_ENDPOINT = "/api/auth/login";
    const AUTH_LOGOUT_ENDPOINT = "/api/auth/logout";
    const AUTH_SESSION_ENDPOINT = "/api/auth/session";

    const body = document.body;
    const form = document.getElementById("sih-auth-form");
    const emailInput = document.getElementById("sih-auth-email");
    const passwordInput = document.getElementById("sih-auth-password");
    const submitButton = document.getElementById("sih-auth-submit");
    const message = document.getElementById("sih-auth-message");
    const userBar = document.getElementById("sih-auth-user-bar");
    const userLabel = document.getElementById("sih-auth-user-label");
    const logoutButton = document.getElementById("sih-auth-header-logout");

    if (!body || !form || !emailInput || !passwordInput || !submitButton) {
        return;
    }

    function setMessage(text) {
        message.textContent = text || "";
    }

    function setAuthenticated(user) {
        body.classList.remove("auth-locked");
        body.classList.add("authenticated");

        if (userBar) {
            userBar.style.display = "block";
        }

        if (userLabel) {
            userLabel.textContent = `${user.username} · ${user.role}`;
        }
    }

    function setUnauthenticated() {
        body.classList.remove("authenticated");
        body.classList.add("auth-locked");

        if (userBar) {
            userBar.style.display = "none";
        }

        if (userLabel) {
            userLabel.textContent = "";
        }
    }

    async function checkSession() {
        try {
            const response = await fetch(
                AUTH_SESSION_ENDPOINT,
                {
                    method: "GET",
                    credentials: "same-origin",
                    cache: "no-store",
                },
            );

            if (!response.ok) {
                setUnauthenticated();
                return false;
            }

            const data = await response.json();

            if (
                data.status !== "authenticated"
                || !data.user
            ) {
                setUnauthenticated();
                return false;
            }

            setAuthenticated(data.user);
            return true;
        } catch (error) {
            setUnauthenticated();
            setMessage(
                "Authentication service is unavailable.",
            );
            return false;
        }
    }

    async function login(event) {
        event.preventDefault();

        const email = emailInput.value.trim();
        const password = passwordInput.value;

        setMessage("");

        if (!email || !password) {
            setMessage(
                "Enter your email and password.",
            );
            return;
        }

        submitButton.disabled = true;
        submitButton.textContent = "Signing In...";

        try {
            const response = await fetch(
                AUTH_LOGIN_ENDPOINT,
                {
                    method: "POST",
                    credentials: "same-origin",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        email,
                        password,
                    }),
                },
            );

            const data = await response.json();

            if (!response.ok) {
                setMessage(
                    data.detail
                    || "Authentication failed.",
                );
                return;
            }

            passwordInput.value = "";

            setAuthenticated(data.user);

            window.location.reload();
        } catch (error) {
            setMessage(
                "Unable to reach the local authentication service.",
            );
        } finally {
            submitButton.disabled = false;
            submitButton.textContent = "Sign In";
        }
    }

    async function logout() {
        try {
            await fetch(
                AUTH_LOGOUT_ENDPOINT,
                {
                    method: "POST",
                    credentials: "same-origin",
                    cache: "no-store",
                },
            );
        } finally {
            setUnauthenticated();
            window.location.reload();
        }
    }

    form.addEventListener("submit", login);

    if (logoutButton) {
        logoutButton.addEventListener(
            "click",
            logout,
        );
    }

    checkSession();
})();
