#############################################################################
##########  "Parfois nous sommes sur le chemin mais on l'ignore    ##########
##########    jusqu'à ce qu'on atteigne notre destination"         ##########
##########                              Brice KENGNI ZANGUIM.      ##########
#############################################################################

#############################################################################
###########           URL de déployement sur Streamlit            ###########
#############################################################################

# 

###################################################################
##########    Importation de bibliothèque utilitaires    ##########
###################################################################

from model_prediction import *
import streamlit as st
import pandas as pd
import numpy as np


############################################################################
##########    paramétrisation des caractéristiques des buttons    ##########
############################################################################
st.title("Recommandation de Livres")
st.markdown("""
<style>
    .stButton button {
        background-color: green ;
        #font-weight: bold;
    }

</style>
""", unsafe_allow_html=True)

########################################################################################
############                    lecture des DataFrames                    ##############
########################################################################################

data = pd.read_csv("DataFrames/data.csv")
articles_metadata = pd.read_csv("DataFrames/articles_metadata.csv")
article_embedding_80 = pd.read_csv("DataFrames/article_embedding_80.csv")
article_embedding_80 = article_embedding_80.set_index( articles_metadata.article_id.values )

with st.sidebar:
   
###############################################################################################################
############################    Afficher un message pour expliquer l'application    ###########################
###############################################################################################################

	st.write("# Interface graphique de test de l'application de recommandation d'articles réalisée par Brice KENGNI ZANGUIM")


#############################################################################################################
#############################     acquisition du nom de modèle à utiliser       #############################
#############################################################################################################

	models = []

	st.write("###  1 - Quels models voudriez vous utiliser  ?")
	st.write("###### plusieurs choix sont possibles")

	content_base = st.checkbox("Content-based Recommandation", True)
	SVD = st.checkbox("Collaborative Filtering - SVD")
	distance = st.checkbox("Embedding - Distance Similarité")
	cosinus = st.checkbox("Embedding - Cosinus Similarité")


#############################################################################################################
###########     Actualisation de la variable models contenant la liste des modèles à utiliser     ###########
#############################################################################################################

	if content_base :
		models.append("Content-based Recommandation")
	if SVD :
		models.append("Collaborative Filtering - SVD")
	if distance :
		models.append("Embedding - Distance Similarité")
	if cosinus :
		models.append("Embedding - Cosinus Similarité")
	
#############################################################################################################
###########        Nombre de recommandation à proposer à l'utilisateur pour chaque modèle         ###########
#############################################################################################################

	st.write("###  2 - Quel est le nombre d'articles à recommander  ?")
	n = st.slider( label ="",
				   min_value = 1, 
				   max_value = 25, 
				   value = 10
				 )

#############################################################################################################
######################          Acquisition de l'identifiant de l'utilisateur          ######################
#############################################################################################################

	user_list = np.unique( data.user_id.values )

	st.write("###  3 - Quel est l'identifiant de l' utilisateur à qui recommander les articles  ?")

	user_id =  st.slider( label ="",
				   min_value = 0, 
				   max_value = len(user_list) -1, 
				   value = 5
				 )

	user_id = user_list[user_id]

	recomand = st.button( "Recommander" )


#######################################################################################
######   Affichage de la liste des articles déjà consultés par l'utilisateur    #######
#######################################################################################


user_articles_list = anciens_articles_utilisateurs( data, articles_metadata, user_id ) 

if len(user_articles_list) == 1:
	st.write("###  A - Article déjà consulté par l'utilisateur")
elif len(user_articles_list) > 1:
	st.write("###  A - Articles déjà consultés par l'utilisateur")

st.write( user_articles_list )




######################################################################################
###########            Effectuation de la prédiction de masque             ###########
###################################################################################### 
if recomand :
	recommandation = prediction(models, data, articles_metadata, article_embedding_80, user_id,  n)

	if len(models) > 0 : 

		#######################################################################################
		###########            Affichage des recommandations d'articles             ###########
		#######################################################################################

		st.write(f"###  B - Recommandations d'articles à l'utilisateur")

		columns = st.columns(len(models))
		for i in range(len(models)) :
			with columns[i] :
				st.write(f"###### {i+1} - {models[i]}")
				st.write(recommandation[i])
		
#st.balloons()

