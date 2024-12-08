from transformers import MBartForConditionalGeneration, MBart50Tokenizer

def traduzir(texto, modelo, tokenizador):
    """
    Função para traduzir um texto do inglês para o português usando o modelo mBART.
    
    Args:
        texto (str): O texto a ser traduzido.
        modelo (MBartForConditionalGeneration): O modelo mBART carregado.
        tokenizador (MBart50Tokenizer): O tokenizador correspondente ao modelo.

    Returns:
        str: O texto traduzido.
    """
    # Adicionar o texto no formato necessário
    texto_com_tag = f"{texto}"
    
    # Tokenizar o texto de entrada (converte o texto em tensores que o modelo entende)
    entrada_tokens = tokenizador(
        texto_com_tag,
        return_tensors="pt",  # Formato PyTorch
        padding=True,         # Adiciona padding para textos curtos
        truncation=True,      # Corta textos longos
        max_length=512        # Limite máximo de 512 tokens
    )
    
    # Gerar a tradução com o modelo
    traduzido_tokens = modelo.generate(
        **entrada_tokens,
        max_length=512,  # Limite máximo para o texto gerado
        forced_bos_token_id=tokenizador.lang_code_to_id["pt_XX"]  # Força o idioma de destino a ser português
    )
    
    # Decodificar os tokens traduzidos de volta para texto
    traducao = tokenizador.decode(traduzido_tokens[0], skip_special_tokens=True)
    return traducao

def texto_usuario():
    """
    Função para capturar e validar a entrada do usuário.
    
    Returns:
        str: Um texto válido fornecido pelo usuário.
    """
    while True:
        try:
            # Capturar a entrada do usuário
            texto = input("Digite o texto para ser traduzido: ")
            
            # Verificar se o texto está vazio
            if not texto:
                print("Erro: Insira um texto para ser traduzido!")
                continue
            
            # Verificar se o texto excede o limite de 512 caracteres
            if len(texto) > 512:
                print("Erro: O texto deve ter no máximo 512 caracteres.")
                continue
            
            return texto  # Retornar o texto válido
        except ValueError:
            print("Erro: Insira um texto válido.")

if __name__ == "__main__":
    """
    Ponto de entrada principal do programa.
    """
    # Identificador do modelo pré-treinado
    modelo_id = "facebook/mbart-large-50-many-to-many-mmt"
    
    # Carrega o tokenizador e o modelo mBART
    print("Carregando modelo e tokenizador...")
    tokenizador = MBart50Tokenizer.from_pretrained(modelo_id)
    modelo = MBartForConditionalGeneration.from_pretrained(modelo_id)

    # Configurar os idiomas de origem e destino
    tokenizador.src_lang = "en_XX"  # Inglês como idioma de origem
    tokenizador.tgt_lang = "pt_XX"  # Português como idioma de destino

    # Capturar o texto do usuário
    texto_original = texto_usuario()

    # Processar e traduzir o texto
    print("\nProcessando Tradução...")
    traducao = traduzir(texto_original, modelo, tokenizador)

    # Exibir os resultados
    print("\n=============================")
    print("Texto Original:", texto_original)
    print("-----------------------------")
    print("Tradução:", traducao)
    print("=============================")

    # Finalizar o programa
    input("Pressione ENTER para sair...")
