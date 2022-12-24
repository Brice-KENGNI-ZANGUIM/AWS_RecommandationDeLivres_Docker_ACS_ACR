FROM python:3.9

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /recommandation_app

COPY requirements.txt ./requirements.txt

# Installation des dépendances
RUN pip3 install -r requirements.txt

# Exposition de l'application sur le port
#ENV PORT 8501
EXPOSE 8501

# copying all files over
COPY . /recommandation_app

# commande à exécuter lorsque le contenaire est lancé
#CMD streamlit run streamlit_main.py
#ENTRYPOINT [ "streamlit", "run", "streamlit_main.py" , "--server.port=8501", "--server.address=127.0.0.1" ]
ENTRYPOINT [ "streamlit", "run" ]
CMD [ "streamlit_main.py"]

# configuration spécifiques à Streamlit
#ENV LC_ALL=C.UTF-8
#ENV LANG=C.UTF-8
#RUN mkdir -p /root/.streamlit
#RUN bash -c 'echo -e "\
#[general]\n\
#email = \"\"\n\
#" > /root/.streamlit/credentials.toml'

#RUN bash -c 'echo -e "\
#"[server]\n\
#enableCORS = True\n\
#" > /root/.streamlit/config.toml'


# Le DockerImage peut être testé avec la commande docker run -p 8501:8501 <Docker Image Name>
