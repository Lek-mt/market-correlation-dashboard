import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.cluster.hierarchy as sch

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Market Correlations | Mathis Turleque", layout="wide")
sns.set_theme(style="darkgrid")

# --- 1. DÉFINITION DES ACTIFS (Nom + Ticker) ---
# Structure : "Catégorie" : {"Ticker": "Nom complet"}

ASSETS = {
    "Géants Tech (US)": {
        'NVDA': 'Nvidia',
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'TSLA': 'Tesla',
        'AMZN': 'Amazon',
        'GOOGL': 'Google'
    },
    "France (CAC 40)": {
        '^FCHI': 'CAC 40 Index',
        'MC.PA': 'LVMH',
        'OR.PA': "L'Oréal",
        'TTE.PA': 'TotalEnergies',
        'AIR.PA': 'Airbus',
        'RMS.PA': 'Hermès'
    },
    "Indices Mondiaux": {
        'SPY': 'S&P 500 (USA)',
        'QQQ': 'Nasdaq 100 (Tech)',
        '^GDAXI': 'DAX 40 (Allemagne)',
        '^N225': 'Nikkei 225 (Japon)'
    },
    "Crypto": {
        'BTC-USD': 'Bitcoin',
        'ETH-USD': 'Ethereum',
        'SOL-USD': 'Solana',
        'DOGE-USD': 'Dogecoin'
    },
    "Valeurs Refuges & Forex": {
        'GLD': 'Or (Gold)',
        'SLV': 'Argent (Silver)',
        'EURUSD=X': 'Euro / Dollar'
    },
    "Secteurs & Énergie": {
        'XLE': 'Énergie (US)',
        'XLF': 'Finance (US)',
        'USO': 'Pétrole (WTI)'
    }
}

# --- CRÉATION DES LISTES TECHNIQUES ---
# 1. La liste simple de tous les tickers pour le téléchargement
ALL_TICKERS = []
# 2. Le dictionnaire de "Traduction" pour l'affichage : {'NVDA': 'Nvidia (NVDA)'}
TICKER_MAP = {}

for category, items in ASSETS.items():
    for ticker, name in items.items():
        ALL_TICKERS.append(ticker)
        TICKER_MAP[ticker] = f"{name} ({ticker})"
ALL_TICKERS = [item for sublist in ASSETS.values() for item in sublist]


# --- 2. FONCTIONS DE CHARGEMENT ---
@st.cache_data
def load_data(tickers, period):
    data = yf.download(tickers, period=period)['Close']
    if data.empty:
        return None, None
    returns = data.pct_change().dropna()
    return data, returns

# --- 3. INTERFACE LATÉRALE ---
st.sidebar.title("⚙️ Paramètres")
st.sidebar.markdown("Panel de contrôle quantitatif")

period_options = ['1y', '2y', '5y']
selected_period = st.sidebar.selectbox("Période d'analyse", period_options, index=2)

with st.sidebar.status(f"Téléchargement des données ({selected_period})..."):
    prices, returns = load_data(ALL_TICKERS, selected_period)

if returns is None:
    st.error("Erreur de téléchargement. Vérifie ta connexion.")
    st.stop()

st.sidebar.success("Données chargées !")

st.sidebar.subheader("Sélection d'actifs")
selected_assets = st.sidebar.multiselect(
    "Choisis les actifs :",
    options=ALL_TICKERS,
   
    format_func=lambda x: TICKER_MAP.get(x, x),
    default=['SPY', 'BTC-USD', 'NVDA', 'GLD', '^FCHI']
)

# --- 4. CORPS PRINCIPAL ---
st.title("📊 Analyseur de Corrélations Cross-Asset")
st.markdown("**Auteur :** Mathis Turleque | *Projet d'analyse quantitative*")

tab1, tab2, tab3, tab4 = st.tabs(["🧩 Clustermap", "🔥 Heatmap", "📈 Rolling", "⚡ Risk/Reward"])

# --- TAB 1 : CLUSTERMAP ---
with tab1:
    st.subheader("Clustering Hiérarchique (Regroupement Intelligent)")
    st.markdown("Les actifs sont **réorganisés** pour rapprocher ceux qui se comportent de la même manière.")
    
    if len(selected_assets) > 2:
        # 1. Calcul de la Matrice de Corrélation
        corr_matrix = returns[selected_assets].corr()
        
        # 2. Le Calcul Savant (Clustering)
        d = sch.distance.pdist(corr_matrix)
        L = sch.linkage(d, method='ward')
        
        # 3. On extrait l'ordre idéal (les "feuilles" de l'arbre)
        dendro = sch.dendrogram(L, no_plot=True)
        ordered_cols = corr_matrix.columns[dendro['leaves']].tolist()
        
        # 4. On réorganise la matrice selon cet ordre
        df_ordered = corr_matrix.loc[ordered_cols, ordered_cols]
        
        # 5. Affichage Interactif avec Plotly
        fig_cluster = px.imshow(
            df_ordered,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale='RdBu_r', 
            zmin=-1, zmax=1,
            origin='lower'
        )
        
        fig_cluster.update_layout(
            title="Heatmap Réorganisée par Similarité",
            xaxis_title="Actifs (Regroupés)",
            yaxis_title="Actifs (Regroupés)",
            width=800, height=800
        )
        
        st.plotly_chart(fig_cluster, use_container_width=True)
        
    else:
        st.warning("Sélectionne au moins 3 actifs pour faire un clustering.")

# --- TAB 2 : HEATMAP INTERACTIVE (PLOTLY) ---
with tab2:
    st.subheader("Matrice de Corrélation Interactive")
    st.markdown("Passe la souris sur les cases pour voir les détails exacts.")
    
    if len(selected_assets) > 1:
        corr_matrix = returns[selected_assets].corr()
        
        # Création de la Heatmap interactive avec Plotly
        fig_heat = px.imshow(
            corr_matrix,
            text_auto=".2f",                
            aspect="auto",                  
            color_continuous_scale='RdBu_r', 
            zmin=-1, zmax=1,                
            origin='lower'                  
        )
        
        # Petite retouche cosmétique pour que ce soit plus joli
        fig_heat.update_layout(
            title="Matrice de Corrélation",
            xaxis_title="Actifs",
            yaxis_title="Actifs",
            width=800,
            height=800
        )
        
        st.plotly_chart(fig_heat, use_container_width=True)
        
    else:
        st.warning("Sélectionne au moins 2 actifs dans la barre latérale.")

# --- TAB 3 : ROLLING CORRELATION (INTERACTIF) ---
with tab3:
    st.subheader("Analyse Dynamique (Rolling Window)")
    
    # Création des colonnes
    col1, col2, col3 = st.columns(3)
    
    
    with col1:
        asset_a = st.selectbox(
            "Actif A", 
            ALL_TICKERS, 
            index=ALL_TICKERS.index('BTC-USD'),
            format_func=lambda x: TICKER_MAP.get(x, x)
        )
    with col2:
        asset_b = st.selectbox(
            "Actif B", 
            ALL_TICKERS, 
            index=ALL_TICKERS.index('QQQ'),
            format_func=lambda x: TICKER_MAP.get(x, x)
        )
    with col3:
        window_days = st.slider("Fenêtre (Jours)", 30, 252, 90)

    
    if asset_a != asset_b:
        # Calcul
        rolling_corr = returns[asset_a].rolling(window=window_days).corr(returns[asset_b])
        
        # --- LA MAGIE PLOTLY COMMENCE ICI ---
        df_chart = rolling_corr.reset_index()
        df_chart.columns = ['Date', 'Corrélation']
        
        # Création du graphique interactif
        fig = px.line(
            df_chart, 
            x='Date', 
            y='Corrélation', 
            title=f"Corrélation {window_days}j : {asset_a} vs {asset_b}",
            color_discrete_sequence=['#4CAF50'] # Couleur verte "Finance"
        )
        
        # Ajout de la ligne zéro
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.update_yaxes(range=[-1.1, 1.1])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistique rapide
        curr_corr = rolling_corr.iloc[-1]
        st.metric(label=f"Corrélation actuelle", value=f"{curr_corr:.2f}")
        
    else:
        st.error("Choisis deux actifs différents.")

        # --- TAB 4 : RISK / REWARD (MARKOWITZ) ---
with tab4:
    st.subheader("Analyse Risque / Rendement (Approche Markowitz)")
    st.markdown("Comparaison de la performance annualisée par rapport à la volatilité (risque).")
    
    if len(selected_assets) > 0:
        # Calcul des métriques annuelles (252 jours de trading)
        daily_returns = returns[selected_assets]
        annual_return = daily_returns.mean() * 252
        annual_volatility = daily_returns.std() * (252 ** 0.5)
        
        # Création d'un DataFrame propre pour Plotly
        risk_return_df = pd.DataFrame({
            'Actif': selected_assets,
            'Rendement Annualisé': annual_return,
            'Volatilité (Risque)': annual_volatility
        })
        
        # Le Scatter Plot Interactif
        fig_risk = px.scatter(
            risk_return_df,
            x='Volatilité (Risque)',
            y='Rendement Annualisé',
            text='Actif', # Affiche le nom sur le point
            size=[15]*len(selected_assets), # Taille des points fixe
            color='Actif', # Une couleur par actif
            title="Frontière Efficiente (Risk vs Reward)"
        )
        
        # Lignes pour diviser le graphique en 4 cadrans
        fig_risk.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
        fig_risk.add_vline(x=annual_volatility.mean(), line_dash="dash", line_color="white", opacity=0.3)
        
        # Mettre les étiquettes (textes) un peu au dessus des points pour lisibilité
        fig_risk.update_traces(textposition='top center')
        
        st.plotly_chart(fig_risk, use_container_width=True)
        
        st.info("💡 **Lecture :** Les meilleurs actifs sont en **Haut à Gauche** (Rendement élevé, Risque faible).")
        
    else:
        st.warning("Sélectionne des actifs pour voir l'analyse.")