const API_BASE = "http://127.0.0.1:5000";

let selectedGenres = [];
let currentMode = "existing"; // "existing" ya "new"

// ---------- Genre-based Poster Colors (fallback jab real poster na mile) ----------
const GENRE_COLORS = {
    "Action": ["#ff5f6d", "#ffc371"],
    "Adventure": ["#f7971e", "#ffd200"],
    "Animation": ["#43cea2", "#185a9d"],
    "Children's": ["#ff9a9e", "#fecfef"],
    "Comedy": ["#ffb347", "#ffcc33"],
    "Crime": ["#414345", "#232526"],
    "Documentary": ["#606c88", "#3f4c6b"],
    "Drama": ["#654ea3", "#eaafc8"],
    "Fantasy": ["#7f00ff", "#e100ff"],
    "Film-Noir": ["#232526", "#414345"],
    "Horror": ["#0f0c29", "#302b63"],
    "Musical": ["#ff6a88", "#ff99ac"],
    "Mystery": ["#4b6cb7", "#182848"],
    "Romance": ["#ff758c", "#ff7eb3"],
    "Sci-Fi": ["#00c6ff", "#0072ff"],
    "Thriller": ["#232526", "#0f2027"],
    "War": ["#485563", "#29323c"],
    "Western": ["#c79081", "#dfa579"],
};
const DEFAULT_GRADIENT = ["#6c5ce7", "#c56cf0"];

function getPosterGradient(genres) {
    const primaryGenre = genres.find(g => GENRE_COLORS[g]);
    const colors = primaryGenre ? GENRE_COLORS[primaryGenre] : DEFAULT_GRADIENT;
    return `linear-gradient(135deg, ${colors[0]}, ${colors[1]})`;
}

// ---------- DOM Elements ----------
const existingUserBtn = document.getElementById("existingUserBtn");
const newUserBtn = document.getElementById("newUserBtn");
const existingUserSection = document.getElementById("existingUserSection");
const newUserSection = document.getElementById("newUserSection");
const userSelect = document.getElementById("userSelect");
const genreChips = document.getElementById("genreChips");
const getRecsBtn = document.getElementById("getRecsBtn");

const loadingSection = document.getElementById("loadingSection");
const errorSection = document.getElementById("errorSection");
const errorMessage = document.getElementById("errorMessage");
const resultsSection = document.getElementById("resultsSection");
const explanationText = document.getElementById("explanationText");
const movieGrid = document.getElementById("movieGrid");

// ---------- Toggle Between Existing / New User ----------
existingUserBtn.addEventListener("click", () => {
    currentMode = "existing";
    existingUserBtn.classList.add("active");
    newUserBtn.classList.remove("active");
    existingUserSection.classList.remove("hidden");
    newUserSection.classList.add("hidden");
});

newUserBtn.addEventListener("click", () => {
    currentMode = "new";
    newUserBtn.classList.add("active");
    existingUserBtn.classList.remove("active");
    newUserSection.classList.remove("hidden");
    existingUserSection.classList.add("hidden");
});

// ---------- Load Users Dropdown ----------
async function loadUsers() {
    try {
        const res = await fetch(`${API_BASE}/api/users`);
        const data = await res.json();

        userSelect.innerHTML = "";
        data.users.forEach(userId => {
            const option = document.createElement("option");
            option.value = userId;
            option.textContent = `User ${userId}`;
            userSelect.appendChild(option);
        });
    } catch (err) {
        userSelect.innerHTML = `<option value="">Failed to load users</option>`;
    }
}

// ---------- Load Genre Chips ----------
async function loadGenres() {
    try {
        const res = await fetch(`${API_BASE}/api/genres`);
        const data = await res.json();

        genreChips.innerHTML = "";
        data.genres.forEach(genre => {
            const chip = document.createElement("span");
            chip.className = "genre-chip";
            chip.textContent = genre;
            chip.addEventListener("click", () => toggleGenre(chip, genre));
            genreChips.appendChild(chip);
        });
    } catch (err) {
        genreChips.innerHTML = `<span class="loading-text">Failed to load genres</span>`;
    }
}

function toggleGenre(chip, genre) {
    if (selectedGenres.includes(genre)) {
        selectedGenres = selectedGenres.filter(g => g !== genre);
        chip.classList.remove("selected");
    } else {
        selectedGenres.push(genre);
        chip.classList.add("selected");
    }
}

// ---------- Get Recommendations ----------
getRecsBtn.addEventListener("click", async () => {
    hideAll();
    loadingSection.classList.remove("hidden");
    getRecsBtn.disabled = true;

    try {
        let data;

        if (currentMode === "existing") {
            const userId = userSelect.value;
            if (!userId) {
                showError("Please select a user.");
                return;
            }
            const res = await fetch(`${API_BASE}/api/recommend/user/${userId}?n=10`);
            data = await res.json();
        } else {
            if (selectedGenres.length === 0) {
                showError("Please select at least one genre.");
                return;
            }
            const res = await fetch(`${API_BASE}/api/recommend/new-user`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ genres: selectedGenres, n: 10 }),
            });
            data = await res.json();
        }

        if (data.error) {
            showError(data.error);
            return;
        }

        showResults(data);

    } catch (err) {
        showError("Could not connect to the server. Please make sure the backend is running.");
    } finally {
        getRecsBtn.disabled = false;
    }
});

// ---------- UI Helpers ----------
function hideAll() {
    loadingSection.classList.add("hidden");
    errorSection.classList.add("hidden");
    resultsSection.classList.add("hidden");
}

function showError(message) {
    hideAll();
    errorMessage.textContent = message;
    errorSection.classList.remove("hidden");
}

const STRATEGY_EXPLANATIONS = {
    "collaborative_filtering": "Based on patterns from users with similar taste to you.",
    "content_based": "Based on genres similar to movies you've rated highly.",
    "popularity_fallback": "These are generally popular, highly-rated movies — we don't have enough data about you yet.",
    "genre_preference": "Based on the genres you selected.",
};

function showResults(data) {
    hideAll();

    const matchedKey = Object.keys(STRATEGY_EXPLANATIONS).find(k => data.strategy.includes(k));
    explanationText.textContent = "Why these movies? " + (STRATEGY_EXPLANATIONS[matchedKey] || "Recommended for you.");

    movieGrid.innerHTML = "";

    if (!data.recommendations || data.recommendations.length === 0) {
        movieGrid.innerHTML = `<p>No recommendations available. Try different genres.</p>`;
        resultsSection.classList.remove("hidden");
        return;
    }

    data.recommendations.forEach((movie, index) => {
        const card = document.createElement("div");
        card.className = "movie-card";

        const poster = document.createElement("div");
        poster.className = "poster";

        if (movie.poster_url) {
            poster.style.backgroundImage = `url(${movie.poster_url})`;
            poster.style.backgroundSize = "cover";
            poster.style.backgroundPosition = "center";
        } else {
            poster.style.background = getPosterGradient(movie.genres);
        }

        const info = document.createElement("div");
        info.className = "movie-info";

        const title = document.createElement("div");
        title.className = "movie-title";
        title.textContent = `${index + 1}. ${movie.title}`;

        const genreWrap = document.createElement("div");
        genreWrap.className = "movie-genres";
        movie.genres.slice(0, 3).forEach(g => {
            const tag = document.createElement("span");
            tag.className = "genre-tag";
            tag.textContent = g;
            genreWrap.appendChild(tag);
        });

        info.appendChild(title);
        info.appendChild(genreWrap);
        card.appendChild(poster);
        card.appendChild(info);
        movieGrid.appendChild(card);
    });

    resultsSection.classList.remove("hidden");
}

// ---------- Init ----------
loadUsers();
loadGenres();