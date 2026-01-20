FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/hf_cache \
    FASTEMBED_CACHE_DIR=/app/hf_cache \
    XDG_CACHE_HOME=/app/hf_cache \
    PW_BROWSERS_PATH=/app/hf_cache/ms-playwright \
    PLAYWRIGHT_BROWSERS_PATH=/app/hf_cache/ms-playwright \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Bootstrap APT + Install base dependencies
RUN set -eux; \
    sed -i -e 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g' \
           -e 's|http://security.ubuntu.com|https://security.ubuntu.com|g' /etc/apt/sources.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        # audio/video processing (transcode, extract frames)
        ffmpeg \
        # system CA store for HTTPS
        ca-certificates \ 
        # verify GPG signatures (repos, files)
        gnupg \
        wget \
        curl \
        unzip \
        python3 \
        python3-pip \
        gcc \
        libpq-dev \
        # OpenCV dependencies
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        # Chrome dependencies
        libasound2 \
        libatk-bridge2.0-0 \
        libatk1.0-0 \
        libatspi2.0-0 \
        libcups2 \
        libdbus-1-3 \
        libdrm2 \
        libgbm1 \
        libgtk-3-0 \
        libnspr4 \
        libnss3 \
        libwayland-client0 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxkbcommon0 \
        libxrandr2 \
        xdg-utils \
        libu2f-udev \
        # Vulkan loader (GPU acceleration)
        libvulkan1 \
        fonts-liberation; \
    # refresh CA store (make sure HTTPS works)
    update-ca-certificates; \
    # clean APT cache to reduce image size
    rm -rf /var/lib/apt/lists/*

# Install Google Chrome
RUN set -eux; \
    wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/googlechrome-linux-keyring.gpg; \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/googlechrome-linux-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" | tee /etc/apt/sources.list.d/google.list; \
    apt-get update; \
    apt-get install -y google-chrome-stable; \
    rm -rf /var/lib/apt/lists/*

# Install ChromeDriver (matching Chrome version)
RUN set -eux; \
    CHROME_VERSION=$(google-chrome --version | awk '{print $3}' | awk -F'.' '{print $1}'); \
    CHROMEDRIVER_VERSION=$(curl -sS "https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_${CHROME_VERSION}"); \
    wget -O /tmp/chromedriver-linux64.zip "https://storage.googleapis.com/chrome-for-testing-public/${CHROMEDRIVER_VERSION}/linux64/chromedriver-linux64.zip"; \
    unzip /tmp/chromedriver-linux64.zip -d /tmp/; \
    mv /tmp/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver; \
    chmod +x /usr/local/bin/chromedriver; \
    rm -rf /tmp/chromedriver*; \
    chromedriver --version

# Install Deno JS runtime (required for yt-dlp YouTube JS challenge solving)
# See: https://github.com/yt-dlp/yt-dlp/wiki/Extractors#po-token-guide
RUN curl -fsSL https://deno.land/install.sh | sh
ENV DENO_INSTALL="/root/.deno"
ENV PATH="${DENO_INSTALL}/bin:${PATH}"

COPY requirements.txt .

# Install Python packages
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip install -U langchain-huggingface && \
    python3 -m pip install selenium==4.15.2 webdriver-manager && \
    # Install yt-dlp nightly for best YouTube compatibility (SABR/JS challenges)
    python3 -m pip install -U --pre "yt-dlp[default]"

# Clean up NVIDIA/CUDA repos to avoid GPG errors (keep your existing approach)
RUN set -eux; \
    rm -f /etc/apt/sources.list.d/*cuda* /etc/apt/sources.list.d/*nvidia* || true; \
    sed -i -e 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g' \
           -e 's|http://security.ubuntu.com|https://security.ubuntu.com|g' /etc/apt/sources.list

# Install Playwright browsers for both standalone playwright and crawl4ai
# Note: crawl4ai has its own browser management that needs separate setup
RUN python3 -m playwright install --with-deps chromium && \
    crawl4ai-setup || echo "crawl4ai-setup not available, skipping"

# Copy application code
COPY . .

# Copy config files
RUN [ -f auth_config_real.yaml ] && cp -f auth_config_real.yaml src/settings/auth_config.yaml || echo "auth_config_real.yaml not found"
RUN [ -f database_config_real.yaml ] && cp -f database_config_real.yaml src/settings/database_config.yaml || echo "database_config_real.yaml not found"
RUN [ -f rabbitmq_config_real.yaml ] && cp -f rabbitmq_config_real.yaml src/settings/rabbitmq/rabbitmq_config.yaml || echo "rabbitmq_config_real.yaml not found"

CMD ["bash"]