import streamlit as st
import pandas as pd
import psycopg2
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
import unicodedata
from collections import Counter, defaultdict
import os
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

# Configuração da página
st.set_page_config(page_title="Análise de Sentimentos", layout="centered")

@st.cache_resource
# Conexão com o banco de dados
def get_data_from_db():
    try:
        conn = psycopg2.connect(
             host=os.getenv("host"),
             database=os.getenv("database"),
             user=os.getenv("user"),
             password=os.getenv("password")
        )
        query = "SELECT * FROM prova.tabela_tcc"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        return None


def generate_wordcloud(texts):
    # Combina todos os textos em um único string
    text = " ".join(texts)

    # Cria a wordcloud sem pré-processamento
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", max_words=200
    ).generate(text)

    return wordcloud


def get_word_frequency(texts, top_n=10):
    # Junta todos os textos
    all_words = " ".join(texts).split()

    # Conta a frequência das palavras
    word_freq = Counter(all_words)

    # Pega as top N palavras mais frequentes
    top_words = dict(
        sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )

    return top_words


def get_processed_word_frequency(texts, top_n=10):
    # Pré-processa e tokeniza todos os textos
    processed_words = []
    for text in texts:
        processed_text = preprocess_text_for_wordcloud(text)
        processed_words.extend(processed_text.split())

    # Conta a frequência das palavras
    word_freq = Counter(processed_words)

    # Pega as top N palavras mais frequentes
    top_words = dict(
        sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )

    return top_words


# Pré-processamento de texto
def preprocess_text(text):
    # Remove acentos
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    # Converte para minúsculas
    text = text.lower()
    # Remove pontuação
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def preprocess_text_for_wordcloud(text):
    # Remove acentos
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )

    # Converte para minúsculas
    text = text.lower()

    # Remove pontuação
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenização
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("portuguese"))
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


def generate_processed_wordcloud(texts):
    # Pré-processa todos os textos
    processed_texts = [preprocess_text_for_wordcloud(text) for text in texts]

    # Combina todos os textos processados
    text = " ".join(processed_texts)

    # Cria a wordcloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=200,
        colormap="viridis",
        prefer_horizontal=0.7,
    ).generate(text)

    return wordcloud


##############


def get_tetragrams(texts, top_n=10):
    # Pré-processa e obtém os tetragramas
    tetragrams = []
    for text in texts:
        # Pré-processamento
        processed_text = preprocess_text_with_custom_stopwords(text)
        words = processed_text.split()

        # Gera tetragramas
        for i in range(len(words) - 3):
            tetragrams.append(tuple(words[i : i + 4]))

    # Conta frequência
    tetragrams_freq = Counter(tetragrams)

    # Retorna os top N mais frequentes
    return dict(
        sorted(tetragrams_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )


def create_tetragrams_graph(texts):
    # Obtém os tetragramas mais frequentes
    top_tetragrams = get_tetragrams(texts)

    # Cria o grafo
    G = nx.DiGraph()

    # Adiciona nós e arestas
    for tetagram, weight in top_tetragrams.items():
        for i in range(3):
            G.add_edge(tetagram[i], tetagram[i + 1], weight=weight)

    return G, top_tetragrams


###############


def preprocess_text_with_custom_stopwords(text):
    # Stopwords personalizadas
    custom_stopwords = {"ja", "so", "pra"}

    # Pega as stopwords padrão do português
    stop_words = set(stopwords.words("portuguese"))

    # Adiciona as stopwords personalizadas
    stop_words.update(custom_stopwords)

    # Remove acentos
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )

    # Converte para minúsculas
    text = text.lower()

    # Remove pontuação
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenização
    tokens = word_tokenize(text)

    # Remove stopwords (incluindo as personalizadas)
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


def generate_custom_wordcloud(texts):
    # Pré-processa todos os textos
    processed_texts = [preprocess_text_with_custom_stopwords(text) for text in texts]

    # Combina todos os textos processados
    text = " ".join(processed_texts)

    # Cria a wordcloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=200,
        colormap="viridis",
        prefer_horizontal=0.7,
    ).generate(text)

    return wordcloud


def get_custom_word_frequency(texts, top_n=10):
    # Pré-processa e tokeniza todos os textos
    processed_words = []
    for text in texts:
        processed_text = preprocess_text_with_custom_stopwords(text)
        processed_words.extend(processed_text.split())

    # Conta a frequência das palavras
    word_freq = Counter(processed_words)

    # Pega as top N palavras mais frequentes
    top_words = dict(
        sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )

    return top_words


# Análise de aspectos
def analyze_aspects(text):
    aspects = {
        "App": ["app", "aplicativo", "tela", "interface", "navegação", "bug"],
        "Crédito": ["credito", "crédito", "limite", "emprestimo", "score"],
        "Atendimento": ["atendimento", "suporte", "chat", "ajuda", "duvida"],
        "Transferência": ["transferencia", "pix", "ted", "pagamento", "saldo"],
        "Taxas": ["taxa", "tarifa", "custo", "cobrança", "juros"],
        "Segurança": ["segurança", "fraude", "golpe", "senha", "bloqueio"],
        "Conta": ["conta", "digital", "cartão", "cadastro", "abertura"],
    }

    text_lower = text.lower()
    found_aspects = defaultdict(int)

    for aspect, keywords in aspects.items():
        for keyword in keywords:
            if keyword in text_lower:
                found_aspects[aspect] += 1

    return dict(found_aspects) if found_aspects else {"Geral": 1}


def main():
    st.title("Análise de Sentimentos - Comentários de Instituições Financeiras")

    # Carrega os dados
    with st.spinner("Carregando dados..."):
        data = get_data_from_db()

    if data is None:
        st.error("Não foi possível carregar os dados.")
        return

    st.write("Aqui estão os dados extraídos do banco de dados:")
    st.write(data)


    # Análise de aspectos
    with st.spinner("Analisando aspectos..."):
        data["aspects"] = [analyze_aspects(comment) for comment in data["comentario"]]

    # Container centralizado para os gráficos
    st.write("## Visualizações dos Resultados")

    # Nuvem de Palavras
    st.write("")
    st.subheader("Nuvem de Palavras dos Comentários")

    # Gera a nuvem de palavras
    wordcloud = generate_wordcloud(data["comentario"])

    # Cria e exibe o gráfico
    fig_wordcloud = plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig_wordcloud)

    # Gráfico de Frequência de Palavras
    st.write("")
    st.subheader("Top 10 Palavras Mais Frequentes")

    # Obtém as palavras mais frequentes
    top_words = get_word_frequency(data["comentario"])

    # Cria o gráfico de barras
    fig_freq = plt.figure(figsize=(12, 6))
    plt.bar(top_words.keys(), top_words.values(), color="skyblue")
    plt.title("Frequência das Palavras Mais Comuns")
    plt.xlabel("Palavras")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45, ha="right")

    # Adiciona os valores sobre as barras
    for i, (word, freq) in enumerate(top_words.items()):
        plt.text(i, freq, str(freq), ha="center", va="bottom")

    plt.tight_layout()
    st.pyplot(fig_freq)

    # Nuvem de Palavras com Pré-processamento
    st.write("")
    st.subheader("Nuvem de Palavras dos Comentários (Com Pré-processamento)")

    # Gera a nuvem de palavras processada
    wordcloud_processed = generate_processed_wordcloud(data["comentario"])

    # Cria e exibe o gráfico
    fig_wordcloud_processed = plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud_processed, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig_wordcloud_processed)

    # Gráfico de Frequência de Palavras Processadas
    st.write("")
    st.subheader("Top 10 Palavras Mais Frequentes (Com Pré-processamento)")

    # Obtém as palavras mais frequentes processadas
    top_words_processed = get_processed_word_frequency(data["comentario"])

    # Cria o gráfico de barras
    fig_freq_processed = plt.figure(figsize=(12, 6))
    plt.bar(
        top_words_processed.keys(), top_words_processed.values(), color="lightgreen"
    )
    plt.title("Frequência das Palavras Mais Comuns (Com Pré-processamento)")
    plt.xlabel("Palavras")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45, ha="right")

    # Adiciona os valores sobre as barras
    for i, (word, freq) in enumerate(top_words_processed.items()):
        plt.text(i, freq, str(freq), ha="center", va="bottom")

    plt.tight_layout()
    st.pyplot(fig_freq_processed)

    st.write("")

    ####

    # Nuvem de Palavras com Stopwords Personalizadas
    st.write("")
    st.subheader("Nuvem de Palavras dos Comentários (Com Stopwords Personalizadas)")

    # Gera a nuvem de palavras
    wordcloud_custom = generate_custom_wordcloud(data["comentario"])

    # Cria e exibe o gráfico
    fig_wordcloud_custom = plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud_custom, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig_wordcloud_custom)

    # Gráfico de Frequência de Palavras
    st.write("")
    st.subheader("Top 10 Palavras Mais Frequentes (Com Stopwords Personalizadas)")

    # Obtém as palavras mais frequentes
    top_words_custom = get_custom_word_frequency(data["comentario"])

    # Cria o gráfico de barras
    fig_freq_custom = plt.figure(figsize=(12, 6))
    plt.bar(top_words_custom.keys(), top_words_custom.values(), color="lightgreen")
    plt.title("Frequência das Palavras Mais Comuns")
    plt.xlabel("Palavras")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45, ha="right")

    # Adiciona os valores sobre as barras
    for i, (word, freq) in enumerate(top_words_custom.items()):
        plt.text(i, freq, str(freq), ha="center", va="bottom")

    plt.tight_layout()
    st.pyplot(fig_freq_custom)


    # Grafo de Tetragramas
    st.write("")
    st.subheader("Grafo de Tetragramas Mais Frequentes")

    # Cria o grafo
    G, top_tetragrams = create_tetragrams_graph(data["comentario"])

    # Configura o layout do grafo
    pos = nx.spring_layout(G, k=3, seed=20, iterations=50)

    # Cria a figura
    fig_graph = plt.figure(figsize=(15, 10))

    # Desenha as arestas com largura baseada no peso
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="gray",
        width=[w / max(edge_weights) * 3 for w in edge_weights],
        alpha=0.5,
        arrows=True,
        arrowsize=20,
    )

    # Desenha os nós
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=2000, alpha=0.7)

    # Adiciona labels aos nós
    nx.draw_networkx_labels(G, pos, font_size=12)

    plt.title("Conexões entre Palavras Frequentes nos Comentários", fontsize=20)
    plt.axis("off")
    plt.tight_layout()

    # Mostra o grafo
    st.pyplot(fig_graph)

    st.write("")
    st.subheader("Frequência dos Tetragramas Mais Comuns")

    top_tetragrams_bar = get_tetragrams(data["comentario"], top_n=20)

    # Prepara e ordena os dados para o gráfico
    sorted_tetragrams = sorted(
        top_tetragrams_bar.items(), key=lambda item: item[1]
    )  # Ordenação crescente
    tetagram_labels = [" → ".join(tetagram) for tetagram, _ in sorted_tetragrams]
    frequencies = [freq for _, freq in sorted_tetragrams]

    # Cria o gráfico de barras horizontais
    fig_tetagram = plt.figure(figsize=(10, 8))
    bars = plt.barh(range(len(frequencies)), frequencies, color="lightblue")

    # Configurações do gráfico
    plt.title(
        "Tetragramas Mais Frequentes nos Comentários", fontsize=16
    )  # Tamanho do título ajustado
    plt.xlabel("Frequência", fontsize=12)
    plt.ylabel("Sequência de Palavras", fontsize=12)

    # Configura os rótulos do eixo Y
    plt.yticks(range(len(tetagram_labels)), tetagram_labels, fontsize=10)

    # Adiciona os valores ao lado das barras
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f"{int(width)}",
            ha="left",
            va="center",
            fontsize=10,
        )

    # Ajusta o layout para evitar cortes
    plt.tight_layout()

    # Mostra o gráfico
    st.pyplot(fig_tetagram)


    ####



    



    ###











    st.write("")
    st.subheader("Análise Detalhada dos Comentários")



    # Cria DataFrame com análise detalhada
    
    



    # Cria DataFrame com a análise




    # Aplica estilo para destacar os sentimentos com cores
    



    

    # Aplica o estilo e mostra a tabela






    # Adiciona filtros




    # Aplica filtros





    # Ordena por score selecionado



    ######

    st.write("")
    # Primeiro gráfico - Distribuição de Sentimentos




    # Espaço entre os gráficos
    st.write("")

    # Segundo gráfico - Distribuição de Scores


    # Espaço entre os gráficos
    st.write("")

    # Terceiro gráfico - Análise de Aspectos
    st.subheader("Análise de Aspectos")
    all_aspects = defaultdict(int)
    for aspects in data["aspects"]:
        for aspect, count in aspects.items():
            all_aspects[aspect] += count

    fig3 = plt.figure(figsize=(10, 6))
    aspect_items = sorted(all_aspects.items(), key=lambda x: x[1], reverse=True)
    aspects, counts = zip(*aspect_items)

    plt.bar(aspects, counts, color="lightblue")
    plt.title("Aspectos Mencionados nos Comentários")
    plt.xlabel("Aspectos")
    plt.ylabel("Número de Menções")
    plt.xticks(rotation=45)

    # Adiciona os valores sobre as barras
    for i, v in enumerate(counts):
        plt.text(i, v, str(v), ha="center", va="bottom")

    plt.tight_layout()
    st.pyplot(fig3)


    # Estatísticas gerais
    #total_comments, positive_perc, negative_perc, neutral_perc

    st.write("")



    # Exemplos de comentários



if __name__ == "__main__":
    main()
