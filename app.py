import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Configuration de la page
st.set_page_config(page_title="Pokémon Investment", layout="wide")
px.defaults.color_continuous_scale = px.colors.qualitative.Plotly
px.defaults.template = "plotly_white"


# Charger les données
@st.cache_data(show_spinner=False)
def load_data():
    converters = {
    'Prix d\'achat Unitaire': lambda x: float(x.replace('€', '').strip()) if x else None,
    'Prix d\'achat total': lambda x: float(x.replace('€', '').strip()) if x else None,
    'Decembre 24': lambda x: float(x.replace('€', '').strip()) if x else None,
    'Janvier 25': lambda x: float(x.replace('€', '').strip()) if x else None,
    'Février 25': lambda x: float(x.replace('€', '').strip()) if x else None,
    'benef%': lambda x: float(x.replace('%', '').strip()) if x else None
    }
    df = pd.read_csv(
        'data/Invest.csv',
        parse_dates=['Date D\'achat'],  # Convertir la colonne de date en datetime
        dayfirst=True,  # Indiquer que le format de date est jour/mois/année
        converters=converters,  # Appliquer les convertisseurs définis
        dtype={
            'Item': 'str',
            'Type': 'str',
            'État': 'str',
            'Quantité': 'int',
            'Enseigne': 'str'
        }
    )
    return df

# 3. Définition des styles
def color_benef(val):
    if val <= -25:
        return 'background-color: #f67a7a'
    elif -25 < val < 0:
        return 'background-color: #f9cb9c'
    elif 0 <= val <= 25:
        return 'background-color: #bfffa2'
    else:
        return 'background-color: #76d64b'

def print_data():
    display_df = st.session_state.df.copy()
    
    # 1. Formatage des dates
    display_df["Date D'achat"] = pd.to_datetime(display_df["Date D'achat"], errors='coerce').dt.strftime('%d/%m/%Y')
    
    # 2. Calcul du benef% avec gestion des erreurs
    last_month = price_columns[-1]
    
    # Calcul sécurisé avec gestion division par zéro
    with np.errstate(divide='ignore', invalid='ignore'):
        display_df['benef%'] = np.where(
            display_df['État'] == 'Ouvert',
            -100.0,  # Forcer en float
            np.where(
                display_df[last_month] == 0,
                0.0,  # Éviter les divisions par zéro
                ((display_df[last_month] - display_df['Prix d\'achat Unitaire']) 
                / display_df[last_month]) * 100
            )
        )
    
    display_df['benef%'] = display_df['benef%'].fillna(0).replace([np.inf, -np.inf], 0).round(2)
    

    # 4. Configuration du style avec Styler.map
    styler = display_df.style \
        .hide(axis="index") \
        .set_properties(**{
            'text-align': 'center',
            'vertical-align': 'middle'
        }) \
        .map(color_benef, subset=['benef%']) \
        .set_properties(subset=['Prix d\'achat total'], 
                      **{'background-color': '#fff2cc'}) \
        .format({
            'benef%': '{:.2f}%',
            'Prix d\'achat Unitaire': '{:.2f}€',
            'Prix d\'achat total': '{:.2f}€',
            **{col: '{:.2f}€' for col in price_columns}
        })
    row_height = 35 # Hauteur estimée d'une ligne
    # max_height = min(800, 50 + len(st.session_state.df) * row_height)
    st.dataframe(styler, use_container_width=True, hide_index=True, height=len(st.session_state.df)* row_height)

# Initialisation session state pour rafraîchissement
if 'df' not in st.session_state:
    st.session_state.df = load_data()

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir une page", ["📊 Visualisation", "⚙️ Gestion d'objets"])

price_columns = [col for col in st.session_state.df.columns 
                    if col not in ['Item', 'Type', 'Date D\'achat', 'État', 
                                  'Prix d\'achat Unitaire', 'Prix d\'achat total',
                                  'Quantité', 'Enseigne', 'benef%']]

# Page de visualisation
if page == "📊 Visualisation":
    # # Bouton de rafraîchissement
    # if st.button("🔄 Rafraîchir les données (cache clear)"):
    #     st.cache_data.clear()
    #     st.session_state.df = load_data()
    #     st.rerun()
        
    # # Vérification des données brutes
    # with st.expander("Debug - Afficher les données brutes"):
    #     st.write("Version des données en mémoire :")
    #     st.write(st.session_state.df[['Item', 'Prix d\'achat total']].head())
        
    #     st.write("Version du fichier Excel :")
    #     fresh_df = pd.read_excel('data/Invest.xlsx', sheet_name='sheet invest', engine='openpyxl')
    #     st.write(fresh_df[['Item', "Prix d\'achat total"]].head())
        
    st.title("Vue globale de la collection")
    
    # Trouver les colonnes de prix dynamiquement
    
    
    # Dernier mois et mois précédent
    last_month = price_columns[-1]
    prev_month = price_columns[-2] if len(price_columns) > 1 else None
    
    # Calcul des indicateurs
    def safe_sum(series):
        return series.fillna(0).sum()
    
    # Investissement Total (constant)
    total_investment = safe_sum(st.session_state.df['Prix d\'achat total'])
    
    # Calculs pour le dernier mois
    df_last = st.session_state.df.fillna({last_month: 0})
    revenue_last = safe_sum(df_last[last_month] * df_last['Quantité'])
    profit_last = revenue_last - total_investment
    profit_pct_last = (profit_last / total_investment) * 100 if total_investment != 0 else 0
    
    # Calculs pour le mois précédent (si existe)
    if prev_month:
        df_prev = st.session_state.df.fillna({prev_month: 0})
        revenue_prev = safe_sum(df_prev[prev_month] * df_prev['Quantité'])
        profit_prev = revenue_prev - total_investment
        profit_pct_prev = (profit_prev / total_investment) * 100 if total_investment != 0 else 0
    
    # Affichage des métriques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Investissement Total", 
            value=f"€{total_investment:,.2f}",
            help="Somme des prix d'achat totaux"
        )
    
    with col2:
        delta_rev = ((revenue_last - revenue_prev)/revenue_prev*100) if prev_month and revenue_prev !=0 else None
        st.metric(
            label="📈 Recette Potentielle", 
            value=f"€{revenue_last:,.2f}",
            delta=f"{delta_rev:.1f}% vs mois précédent" if delta_rev is not None else None,
            help=f"Valeur totale au prix de {last_month}"
        )
    
    with col3:
        delta_profit = ((profit_last - profit_prev)/abs(profit_prev)*100 if prev_month and profit_prev !=0 else None)
        st.metric(
            label="🤑 Bénéfice", 
            value=f"€{profit_last:,.2f}",
            delta=f"{delta_profit:.1f}% vs mois précédent" if delta_profit is not None else None,
            help="Recette - Investissement Total"
        )
        
    with col4:
        delta_pct = (profit_pct_last - profit_pct_prev) if prev_month else None
        st.metric(
            label="📉 % Bénéfice", 
            value=f"{profit_pct_last:.1f}%",
            delta=f"{delta_pct:.1f} points vs mois précédent" if delta_pct is not None else None,
            help="Pourcentage de bénéfice global"
        )
    
    
    st.divider()
    col_1, col_3 = st.columns([3, 2], border=False)
    with col_1:
        st.subheader("Évolution des prix par objet")
    with col_3:
    # Récupération des types disponibles
        types_disponibles = sorted(st.session_state.df['Type'].unique().tolist())
    
        # Widget de filtrage
        types_selectionnes = st.multiselect(
            "🔍 Filtrer par type d'objet",
            options=types_disponibles,
            default=None,
            placeholder="Sélectionnez un ou plusieurs types..."
        )
    
    # Filtrage des données
    if types_selectionnes:
        df_filtre = st.session_state.df[st.session_state.df['Type'].isin(types_selectionnes)]
    else:
        df_filtre = st.session_state.df.copy()

    # Préparation des données pour le graphique
    price_columns = [col for col in df_filtre.columns 
                    if col not in ['Item', 'Type', 'Date D\'achat', 'État', 
                                  'Prix d\'achat Unitaire', 'Prix d\'achat total',
                                  'Quantité', 'Enseigne', 'benef%']]

    df_melted = df_filtre.melt(
        id_vars=['Item', 'Type'], 
        value_vars=price_columns,
        var_name='Mois', 
        value_name='Prix'
    )
    
    # Conversion en catégorie ordonnée
    df_melted['Mois'] = pd.Categorical(df_melted['Mois'], 
                                     categories=price_columns, 
                                     ordered=True)
    
    # Création du graphique interactif
    fig = px.line(
        df_melted.sort_values('Mois'),
        x='Mois',
        y='Prix',
        color='Item',
        markers=True,
        title=f"Évolution des prix ({len(df_filtre)} objets sélectionnés)",
        category_orders={"Mois": price_columns}
    )
    
    # Améliorations du layout
    fig.update_layout(
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            title_text=''
        ),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Dans la page de visualisation (après le graphique existant)
    st.divider()

    # Création des colonnes pour les nouveaux graphiques
    col1, col2, col3= st.columns([2,1,2])

    with col1:
        # Graphique 1 : Bénéfice par type (donut)
        st.subheader("Répartition des bénéfices par type")
        
        # Calcul du bénéfice total par type
        df_profit = st.session_state.df.copy()
        df_profit['benefice'] = (df_profit[last_month] - df_profit['Prix d\'achat Unitaire']) * df_profit['Quantité']
        profit_par_type = df_profit.groupby('Type')['benefice'].sum().reset_index()

        data = profit_par_type.sort_values('benefice', ascending=False)
        fig1 = px.pie(
            data_frame=data,
            names='Type',
            values='benefice',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        # Désactiver le tri automatique
        fig1.update_traces(
            sort=False,
            textposition='inside',
            textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>Bénéfice: €%{value:,.2f}<extra></extra>"
        )
        fig1.update_layout(
            uniformtext_minsize=10,
            margin=dict(t=30, b=10)
        )
        
        st.plotly_chart(fig1, use_container_width=True)

    # with col2:
    # Graphique 1 : Bénéfice par type (donut)
        # Graphique 2 : Évolution des prix par type
        # st.subheader("Évolution moyenne par type")
        
        # # Calcul de l'évolution en %
        # if len(price_columns) >= 2:
        #     df_evo = st.session_state.df.melt(
        #         id_vars=['Type'],
        #         value_vars=[price_columns[0], price_columns[-1]],
        #         var_name='Mois',
        #         value_name='Prix'
        #     )
            
        #     # Calcul de la variation
        #     evolution = df_evo.groupby(['Type', 'Mois'])['Prix'].mean().unstack()
        #     evolution['Évolution (%)'] = ((evolution[price_columns[-1]] - evolution[price_columns[0]]) / evolution[price_columns[0]] * 100).round(1)
            
        #     # Tri des valeurs
        #     evolution = evolution.sort_values('Évolution (%)', ascending=False).head(10)
            
        #     # Création du graphique
        #     fig2 = px.bar(
        #         evolution,
        #         x='Évolution (%)',
        #         y=evolution.index,
        #         orientation='h',
        #         color=evolution.index,
        #         text='Évolution (%)',
        #         color_discrete_sequence=px.colors.qualitative.Dark24
        #     )
            
        #     # Personnalisation
        #     fig2.update_layout(
        #         showlegend=False,
        #         yaxis_title=None,
        #         xaxis_title="Évolution en pourcentage",
        #         margin=dict(t=30, b=10)
        #     )
        #     fig2.update_traces(
        #         texttemplate='%{text}%',
        #         textposition='outside'
        #     )
            
        #     st.plotly_chart(fig2, use_container_width=True)
        # else:
        #     st.warning("Pas assez de données historiques pour calculer l'évolution")
    
    st.divider()
    
    st.subheader("Données de la collection")
    # Création d'une copie pour les modifications d'affichage
    print_data()

# Page de gestion
elif page == "⚙️ Gestion d'objets":
    st.title("Gestion de la collection")
    
    # Onglets Ajouter/Supprimer
    gestion_tab, supprimer_tab = st.tabs(["➕ Ajouter un objet", "🗑️ Supprimer un objet"])
    
    with gestion_tab:
        with st.form("add_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                item = st.text_input("Nom de l'objet*")
                item_type = st.selectbox("Type*", [
                    "Display FR", "Display JAP", "Coffret", "Duo/Tripack",
                    "Bundle", "ETB", "Carte FR", "Carte Jap", "Blister",
                    "Pokébox", "UPC"
                ])
                date_achat = st.date_input("Date d'achat")
                etat = st.selectbox("État*", ["En stock", "Échangé", "Ouvert", "Hors stock"])
            
            with col2:
                prix_unitaire = st.number_input("Prix unitaire (€)*", min_value=0.0, step=0.01)
                quantite = st.number_input("Quantité*", min_value=1, step=1)
                enseigne = st.text_input("Enseigne")
                dernier_prix = st.number_input("Prix Février 25 (€)*", min_value=0.0, step=0.01)
            
            submitted = st.form_submit_button("Valider l'ajout")
            
            if submitted:
                if not item or not item_type or not etat:
                    st.error("Les champs marqués d'un * sont obligatoires")
                else:
                    # Calculs automatiques
                    prix_total = prix_unitaire * quantite
                    benef = ((dernier_prix - prix_unitaire)/dernier_prix) if dernier_prix != 0 else -1
                    
                    # Création de la nouvelle entrée
                    new_entry = {
                        'Item': item,
                        'Type': item_type,
                        'Date D\'achat': date_achat,
                        'État': etat,
                        'Prix d\'achat Unitaire': prix_unitaire,
                        'Prix d\'achat total': prix_total,
                        'Quantité': quantite,
                        'Enseigne': enseigne,
                        'Février 25': dernier_prix,
                        'benef%': benef
                    }
                    
                    # Mise à jour du DataFrame
                    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_entry])], ignore_index=True)
                    st.session_state.df.to_excel('data/Invest.xlsx', index=False, sheet_name='sheet invest')
                    st.success("Objet ajouté avec succès !")
    st.divider()       
    print_data()

    with supprimer_tab:
        st.subheader("Sélectionner un objet à supprimer")
        
        # Création du dataframe réduit
        df_reduit = st.session_state.df[['Item', 'Type', 'Quantité', 'Enseigne', 'Prix d\'achat total']].copy()
        df_reduit['Supprimer'] = False
        
        # Éditeur de données avec boutons de suppression
        edited_df = st.data_editor(
            df_reduit,
            column_config={
                "Supprimer": st.column_config.CheckboxColumn(
                    "Confirmer suppression",
                    help="Cocher la case pour supprimer l'objet",
                    default=False
                )
            },
            disabled=["Item", "Type", "Quantité", "Enseigne", "Prix d\'achat total"],
            use_container_width=True
        )
        
        # Récupération des lignes à supprimer
        to_delete = edited_df[edited_df.Supprimer]
        
        if not to_delete.empty:
            if st.button("Confirmer la suppression définitive"):
                # Filtrage des index à conserver
                keep_index = [i for i in st.session_state.df.index if i not in to_delete.index]
                st.session_state.df = st.session_state.df.loc[keep_index]
                
                # Sauvegarde
                st.session_state.df.to_excel('data/Invest.xlsx', index=False, sheet_name='sheet invest')
                st.rerun()