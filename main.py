import os
from cognitive.langchain_document_adapter import LangchainDocumentAdapter as LDA
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from dotenv import load_dotenv
import nest_asyncio
load_dotenv(dotenv_path='.env')
nest_asyncio.apply()

openai_api_type = os.getenv('OPENAI_API_TYPE')
openai_api_version = os.getenv('OPENAI_API_VERSION')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')


if __name__ == '__main__':
    print("Starting the synthetic data generation process")
    os.environ["OPENAI_API_TYPE"] = openai_api_type
    os.environ["OPENAI_API_VERSION"] = openai_api_version
    os.environ["AZURE_OPENAI_ENDPOINT"] = azure_openai_endpoint
    os.environ["AZURE_OPENAI_API_KEY"] = azure_openai_api_key

    base_model = AzureChatOpenAI(
        model="gpt-35-turbo",
        azure_endpoint=azure_openai_endpoint,
    )
    azure_critic_model = AzureChatOpenAI(
        model="gpt-35-turbo",
        azure_endpoint=azure_openai_endpoint,
    )
    azure_embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-large",
        model="text-embedding-3-large",
        azure_endpoint=azure_openai_endpoint,
    )

    generator = TestsetGenerator.from_langchain(
        generator_llm=base_model,
        critic_llm=azure_critic_model,
        embeddings=azure_embeddings
    )

    generator.adapt(language="magyar", evolutions=[simple, reasoning, multi_context], cache_dir=".cache")

    generator.save(evolutions=[simple, reasoning, multi_context], cache_dir=".cache")

    load_langchain_document = LDA("C:/Work/Telekom/SyntheticData/docs/Az adatvédelmi hatásvizsgálat és előzetes konzultációja_page_6.pdf")

    document = load_langchain_document.read()

    testset = generator.generate_with_langchain_docs(
            documents=document,
            test_size=10,
            distributions={
                simple: 0.5,
                reasoning: 0.25,
                multi_context: 0.25,
            },
            with_debugging_logs=True,
            raise_exceptions=False,
            is_async=False
    )

    df = testset.to_pandas()
    list_of_dicts = df.to_dict(orient='records')
    # print the data
    print(f"Save testset as pandas dataframe")

    testset.to_pandas().to_csv("C:/Work/Telekom/SyntheticData/docs/synthetic_data.csv")
