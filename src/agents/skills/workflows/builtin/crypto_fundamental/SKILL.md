---
name: crypto-fundamental
description: Comprehensive cryptocurrency fundamental analysis including tokenomics, on-chain metrics, ecosystem assessment, and investment thesis. Triggers for crypto deep dive, tokenomics, on-chain analysis.
triggers:
  - tokenomics
  - on-chain
  - crypto fundamental
  - token analysis
  - cryptocurrency analysis
  - phân tích cơ bản crypto
  - tokenomics
tools_hint:
  - getCryptoPrice
  - getCryptoMarketData
  - getCryptoNews
  - webSearch
max_tokens: 1800
---

# Crypto Fundamental Analysis Workflow

Follow these steps for comprehensive cryptocurrency analysis.

## Step 1: Gather Market Data

Call these tools IN PARALLEL:
- `getCryptoPrice(symbol="[TOKEN]")` → Current price, market cap, volume
- `getCryptoMarketData(symbol="[TOKEN]")` → Extended market metrics (if available)
- `getCryptoNews(symbol="[TOKEN]")` → Recent news and sentiment

## Step 2: Tokenomics Analysis

Assess the token's economic design:
- **Supply metrics:** Circulating vs. Total vs. Max supply
- **Inflation rate:** Annual token emission schedule
- **Distribution:** Team/Foundation/Community/Treasury allocation
- **Vesting schedules:** Upcoming unlocks that may create sell pressure
- **Token utility:** Governance, staking, fee payment, collateral

If internal tools lack tokenomics data, use `webSearch` to find:
- Token unlock schedules
- Staking yields and participation rates
- Revenue/fee generation metrics

## Step 3: On-Chain Metrics (if data available)

Key metrics to assess:
- **Active addresses:** Growth trend (bullish) or decline (bearish)
- **Transaction volume:** Real usage vs. wash trading
- **TVL (for DeFi):** Total Value Locked trend
- **Developer activity:** GitHub commits, protocol updates
- **Exchange flows:** Net inflows (bearish) vs. outflows (bullish)

Use `webSearch` for on-chain data if not in internal tools.

## Step 4: Competitive Positioning

- **Category peers:** Compare within same category (L1, DeFi, NFT, etc.)
- **Market cap ranking:** Relative position and trend
- **Key differentiators:** Technology, team, ecosystem, partnerships
- **Risks:** Regulatory, competition, technology risks

## Step 5: Scenario Analysis

Present three scenarios:
- **Bull case:** Catalyst-driven upside (adoption, partnerships, upgrades)
- **Base case:** Continuation of current trends
- **Bear case:** Risk factors materialize (regulation, competition, exploit)

Include price targets or ranges for each scenario if possible.

## Step 6: Present Results

Structure your response:
1. **Token Overview:** Price, market cap, rank, 24h volume
2. **Tokenomics Summary:** Supply, inflation, utility, upcoming unlocks
3. **On-Chain Health:** Key metrics with trend interpretation
4. **Competitive Position:** vs. peers with specific comparisons
5. **Scenario Analysis:** Bull/Base/Bear with catalysts
6. **Investment Thesis:** Rating with key reasoning
7. **Key Risks:** Ranked by probability and impact

**Language:** Match the user's language throughout.
