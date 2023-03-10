########################################################################################
########################################################################################

import pandas as pd
import numpy as np
from sklearn import preprocessing

########################################################################################
##########    Recommandation d'articles selon le modèle content-based      #############
########################################################################################
def content_base_recommandation_par_utilisateur( user_id, rating_list, articles_list, n , anciens_articles   ) :
    """
    Description :
    -------------
        La fonction permet, pour un utilisateur donné, de faire de la recommandation d'articles sur la base des articles déjà 
        connus de ce même utilisateur
    
    Paramètres :
    ------------
        - user_id : int
        ----------
            identifiant de l'utilisateur
            
        - rating_list : pd.Dataframe
        --------------
            DataFrame contenant la liste des utilisateurs("user_id"), la liste des categoris de livres qu'il a lu ("category_id"),
            et les notes associés à chaque catégories ( "Rating")
            
        - articles_list : pd.Dataframe
        ----------------
            DataFrame contenant les metadata de chaque livres : "article_id",  "category_id"
            
        - n : int
        ----
            Nombre de livres à recommander
            
       - anciens_articles : array
        -------------------
            liste des anciens articles déjà consultés par l'uitlisateur
            
            
    Output :  DataFrame 
    ---------
        Dataframe avec les livres recommandés avec les catégories associées
    
    """
    
    _rating_list = rating_list.copy()
    _articles_list = articles_list.copy()[["article_id","category_id"]]
    _user_id = user_id
    
    #  liste des ratings pour chaque catégories pour l'utilisateur
    _user_rating = _rating_list.loc[_rating_list.user_id == _user_id,"Rating"].values[0]
    
    #  liste des catégories de livres  pour l'utilisateur
    _user_categorys = _rating_list.loc[_rating_list.user_id == _user_id, "category_id"].values[0]
    
    # 
    _articles_list["Rating"] = _articles_list.category_id.apply( lambda x : 0 if x not in _user_categorys
                                                                           else 
                                                                            _user_rating[ _user_categorys  == x]
                                                                )
    
    # Selection des livres qui se trouvent dans la même catégorie que celles déjà parcourures par l'utilisateur
    _articles_list = _articles_list[_articles_list.Rating != 0]
    
    # Tris dans l'ordre décroissant des notes de chaque livre
    _articles_list = _articles_list.sort_values(by= ["Rating"], ascending = False)

    # Suppression des articles déjà consultés par l'uilisateur dans la liste des articles à recommander
    _articles_list = _articles_list.loc[ _articles_list.article_id.apply( lambda x : x not in anciens_articles ).values ,:]

    # limitation du nombre de recommandation
    _articles_list = _articles_list.iloc[:n, : ]
    
    # duplicatats
    _articles_list.drop_duplicates("article_id", keep = "first", inplace = True, ignore_index=True)
    
    return _articles_list[["article_id","category_id", "Rating"]]


########################################################################################
################           Classe pour réaliser un LabelEncode            ##############
########################################################################################
class LabelEncode() :
    """
    Description :
    -------------
    	La classe regroupe des méthodes necessaires pour un labelEncode
    
    Méthodes :
    ---------
    	 - fit() :
    	 - transform()
    	 - fit_transform()
    	 - inverse_transform()
 
    """
    
    def __init__ (self , ) :
        
        pass
    
    def fit(self, labels) :
        """
		Description :
		-------------
			fit l'Encodeur sur des données
		
		Paramètres :
		------------
			- labels : list, array ou toute sequence
			----------
				liste d'élements à transformer
				
		Output :  None
		--------
		
		"""
		
        initial = np.unique( labels )
        initial.sort()
        self.encodeur = { k:v for v,k in enumerate(initial) }
        self.decodeur = { k:v for k,v in enumerate(initial) }
        
    def transform (self, labels ) :
        """
		Description :
		-------------
			La méthode éffectue une transformation des données
		
		Paramètres :
		------------
			- labels : list, array ou toute sequence
			----------
				liste d'élements à transformer
		
		Output : array
		--------
				LabelEncode des données
		"""
        transformer = np.vectorize( lambda x : self.encodeur[x] )
        return transformer(labels)
    
    def fit_transform(  self, labels ):
        """
		Description :
		-------------
			fit et transforme les données
		
		Paramètres :
		------------
			- labels : list, array ou toute sequence
			----------
				liste d'élements à transformer
		
		Output :  array
		--------
		
		"""
        self.fit( labels )
        return self.transform(labels)
        
    def inverse_transform(self, labels) :
        """
		Description :
		-------------
			La classe regroupe des méthodes necessaires pour un labelEncode
		
		Paramètres :
		------------
			- labels : list, array ou toute sequence
			----------
				liste d'élements sur lesquels réaliser la transformation inverse
		
		Output :  array
		--------
				Encodage inverse des données
		"""
        transformer = np.vectorize( lambda x : self.decodeur[x] )
        return transformer(labels)


########################################################################################
#########    Classe qui réalise une décomposition en valeurs singulières     ###########
########################################################################################
class SVD:
    """
    Description :
    -------------
    	Classe permettant d'éffectuer de la recommandation d'articles en utilisant la 
    	Décomposition en Valeurs Singulières ( SVD )
    
    méthodes :
    ----------
    	- fit() : Réalise la décomposition de la matrice
    	- predict() : 
    	- recommend() : 
    
    """
    
    def __init__(self, user_mean, metadata ):
        """
        Description :
        -------------
        	Initialisation de l'instance
        
        Paramètres :
        ------------
            - umean : list, tuple, array
            --------
            	Rating moyen pour chaque utilisateur
            
        """
        self.umean = user_mean
        self.metadata = metadata
        
        # init svd resultant matrices
        self.P = np.array([])
        self.S = np.array([])
        self.Qh = np.array([])
        
        # init users and items latent factors
        self.u_factors = np.array([])
        self.i_factors = np.array([])
    	
    def fit(self, R):
        """

        Réalise le fit de la matrice ( Décomposition en 3 Matrices )
        
        Paramètres :
        ------------
        	- R : DataFrame
        	----
        		Matrice à décomposer suivant la méthode SVD
        		
        Output : None
        -------
        
        """
        # Encodeurs
        self.users_encodeur = LabelEncode()
        self.users_encodeur.fit(R.index.values)
        self.items_encodeur = LabelEncode()
        self.items_encodeur.fit(R.columns.values)
        
        
        R = R.values
        P, s, Qh = np.linalg.svd(R, full_matrices=False)
        
        self.P = P
        self.S = np.diag(s)
        self.Qh = Qh
        
        # latent factors of users (u_factors) and items (i_factors)
        self.u_factors = np.dot(self.P, np.sqrt(self.S))
        self.i_factors = np.dot(np.sqrt(self.S), self.Qh)
    
    def predict(self, user_id, item_id):
        """
        Prédit le rating pour un utilisateur et un item donné
        
        Paramètres :
        ------------
            - user_id : int
            ---------
            		identifiant de l'utilisateur
            		
            - item_id : int 
            ----------
            		identifiant de l'item
        
        Output :
        --------
            - r_hat : Rating prédits
        """
        # encode user and item ids
        u = self.users_encodeur.transform([user_id])[0]
        i = self.items_encodeur.transform([item_id])[0]
        
        # the predicted rating is the dot product between the uth row 
        # of u_factors and the ith column of i_factors
        r_hat = np.dot(self.u_factors[u,:], self.i_factors[:,i])
        
        # add the mean rating of user u to the predicted value
        r_hat += self.umean[u]
        
        return r_hat
        
    
    def recommend(self, user_id, n, anciens_articles ):
        """
        
        Description :
        -------------
        	Réalise la recommandation des articles pour un utilisateur donné
        
        Paramètres :
        -----------
            - user_id : int
            ---------
            	Identifiant utilisateur
            	
        Output : DataFrame
        --------
        	DataFrame des articles recommandés par l'algorithme
            	
        """
        # Label encode de l'identifiant de l'utilisateur
        u = self.users_encodeur.transform([user_id])[0]
        
        # the dot product between the uth row of u_factors and i_factors returns
        # the predicted value for user u on all items        
        predictions = np.dot(self.u_factors[u,:], self.i_factors) + self.umean[u]
        
        # sort item ids in decreasing order of predictions
        top_idx = np.flip(np.argsort(predictions))

        # decode indices to get their corresponding itemids
        top_items = self.items_encodeur.inverse_transform(top_idx)
        
        # Suppression des articles déjà consultés par l'utilisateur
        filtre = np.vectorize( lambda x : x not in anciens_articles )
        top_items = top_items[ filtre( top_items ) ]
        
        # recherche les catégories correspondantes aux articles
        #top_categs = [ self.metadata[self.metadata.article_id == art].category_id.values[0] for art in top_items ]
        
        # Ratings rangés par ordre décroissant
        #preds = np.round( predictions[top_idx] , 4)
        
        #return pd.DataFrame({"article_id":top_items, "category_id": top_categs, "Rating":preds })
        return pd.DataFrame({"article_id":top_items})[:n] #pd.DataFrame({"article_id":top_items, "category_id": top_categs, "Rating":preds })[:n]


##########################################################################################
################           Construis les Rating de chaque article           ##############
##########################################################################################
def build_rating( data, label ) :
    """
    Description :
    -------------
    	Construis les ratings
    Paramètres :
    ------------
    	 - data : DataFrame
    	 -------
    	 	DataFrame des clicks utilisateurs
    	 	
    	 - label : str
    	 --------
    	 	chaine de caractères qui représente la colonne à utiliser pour réaliser le Rating.
    	 	Les valeurs permises sont "category" et "article" 
    	 	
    Output :  DataFrame
    ---------
    	DataFrame contenant les Rating de chaque articles ou groupe d'articles pour chaque utilisateur
    
    """
    _df = data.copy()
    
    if "category" in label :
        _data_0 = _df.groupby(["user_id",label]).sum().reset_index()
        _data = _df.groupby(["user_id",label]).size().to_frame().rename(columns = {0:"Rating"}).reset_index()
        _data.Rating = _data.Rating.values * np.round(np.log10( 3 + _data_0.words_count.values/600 ),4)
    elif "article" in label :
        _data_0 = _df.groupby([label]).sum().reset_index()
        _data = _df.groupby([label]).size().to_frame().rename(columns = {0:"Rating"}).reset_index()
        _data.Rating = _data.Rating.values * np.round(np.log10( 3 + _data_0.words_count.values/600 ),4)
    
    return  _data



######################################################################################
################          Construis et process la rating matrix         ##############
######################################################################################
def process_rating_matrix( data, user_mean ) : 
    """
    Description :
    -------------
    	Construis la rating matrix  à partir des données utilisateur et réalise un traitement des valeurs manquantes
    
    Paramètres :
    ------------
    
    Output : 
    ---------
    
    """
    """
    process de la rating_matrix en
    - remplaçant les valeurs manquantes par des zéros
    - normalisant
    """
    _data = data.copy()
    
    _cross = pd.crosstab( _data.iloc[:,0], _data.iloc[:,1], _data.iloc[:,2], aggfunc=sum )
    
    #  Remplacement par des zéros
    _cross = _cross.fillna( 0 )
    #_cross = _cross.fillna( _cross.mean(axis=0) )

    #  Normalisation
    idx, col = _cross.index.values , _cross.columns.values
    _cross = preprocessing.MinMaxScaler().fit_transform(_cross)
    #_cross = _cross.subtract( user_mean , axis = 0 )
    
    return pd.DataFrame( _cross, index=idx , columns=col )


##########################################################################################################
################     Recommande les articles sur la base de la similarité par distance      ##############
##########################################################################################################
def distance_similarité( user_id , user_articles , embedding , article_metadata, n = 10 ) :
    """
    Description :
    -------------
    	Réalise  des recommandations sur la base de la similarité de distance
    	Dans l'espace des embeddings des articles, les articles qui sont les plus proches de l'article ( distance euclidienne )
    	sont les plus succeptibles d'être recommandés à l'utilisateur.
    
    Paramètres :
    ------------
    	- user_id : int 
    	----------
    		identifiant de l'utilisateur
    	
    	- user_articles : DataFrame
    	-----------------
    		DataFrame contenant la liste de de tous les clicks utilisateurs et toutes les informations sur l'article visité, sa catégorie . . .
    		
    	- embedding : DataFrame
    	------------
    		DataFrame contenantles embeddings de tous les articles apparaissant dans le DataFrame "user_articles"
    	
    	- article_metadata : DataFrame
    	--------------------
    		DataFrame contenant toutes les métadata de chaque article.
    		L'ordre des articles dans ce DataFrame et dans le DataFrame "embedding" doit être le même
    		
    	- n : int ; Default = 10
    	-----
    		Nombre d'articles à recommander
    		
    Output : DataFrame 
    --------
    	DataFrame contenant la liste de tous les articles recommandés dans l'ordre croissant de leur similarité de distance
    
    """
    _user_articles = user_articles.copy()
    _embedding = embedding.copy()
    _article_metadata = article_metadata.copy()
    
    # Liste des articles visités par l'utilisateur
    _articles_list = _user_articles[_user_articles.user_id == user_id ].article_id.values
    
    # Calcul des similitudes
    _art , _simil = [] , []
    for _user_article in _articles_list :
        x = _embedding.loc[_user_article,:]
        for _article in _embedding.index.values :
            if _user_article != _article :
                y = _embedding.loc[_article,:]
                _art.append( _article )
                _simil.append( np.linalg.norm(x-y , 2))
    
    # Association des catégories
    _categ = [ _article_metadata[_article_metadata.article_id == art].category_id.values[0] for art in _art ]
    
    # Dataframe
    df = pd.DataFrame({
                        "article_id" : _art , 
                        "category_id" : _categ,
                        "distance_simil" : np.round(_simil , 4)
                      }).sort_values(by = "distance_simil", ascending=True)    
    
    return df.iloc[:n,:]


##########################################################################################################
################     Recommande les articles sur la base de la similarité par cosinus       ##############
##########################################################################################################
def cosinus_similarité( user_id , user_articles , embedding , article_metadata, n = 10 ) :
    """
    Description :
    -------------
    	Réalise  des recommandations sur la base de la similarité de cosinus
    	Dans l'espace des embeddings des articles, les articles qui sont les plus proches de l'article ( distance euclidienne )
    	sont les plus succeptibles d'être recommandés à l'utilisateur.
    
    Paramètres :
    ------------
    	- user_id : int 
    	----------
    		identifiant de l'utilisateur
    	
    	- user_articles : DataFrame
    	-----------------
    		DataFrame contenant la liste de de tous les clicks utilisateurs et toutes les informations sur l'article visité, sa catégorie . . .
    		
    	- embedding : DataFrame
    	------------
    		DataFrame contenantles embeddings de tous les articles apparaissant dans le DataFrame "user_articles"
    	
    	- article_metadata : DataFrame
    	--------------------
    		DataFrame contenant toutes les métadata de chaque article.
    		L'ordre des articles dans ce DataFrame et dans le DataFrame "embedding" doit être le même
    		
    	- n : int ; Default = 10
    	-----
    		Nombre d'articles à recommander
    		
    Output : DataFrame 
    --------
    	DataFrame contenant la liste de tous les articles recommandés dans l'ordre décroissant de leur similarité de cosinus
    
    """
    _user_articles = user_articles.copy()
    _embedding = embedding.copy()
    _article_metadata = article_metadata
    
    # Liste des articles visités par l'utilisateur
    _articles_list = _user_articles[_user_articles.user_id == user_id ].article_id.values
    
    # Calcul des similitudes
    _art , _simil = [] , []
    for _user_article in _articles_list :
        x = _embedding.loc[_user_article,:]
        for _article in _embedding.index.values :
            if _user_article != _article :
                y = _embedding.loc[_article,:]
                _art.append( _article )
                _simil.append(  np.dot(x,y)/(np.linalg.norm(x,2)*np.linalg.norm(y,2) ) )
    
    # Association des catégories
    _categ = [ _article_metadata[_article_metadata.article_id == art].category_id.values[0] for art in _art ]
    
    # Dataframe
    df = pd.DataFrame({
                        "article_id" : _art , 
                        "category_id" : _categ,
                        "cosinus_simil" : np.round(_simil , 4)
                      }).sort_values(by = "cosinus_simil", ascending=False)    
    
    return df.iloc[:n,:]

########################################################################################


