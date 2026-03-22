import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
from utils.config_loader import load_config
from langchain_groq import ChatGroq
from logger.customlogger import CustomLogger
from expection.customExpection import AutoML_Exception

log = CustomLogger().get_logger(__name__)

class ModelLoader:
    def __init__(self) -> None:
        load_dotenv()
        self._validate_env()
        self.config = load_config()
        log.info("Configurations loaded successfully",config_keys = list(self.config.keys()))

    def _validate_env(self):
        required_varibale = ["GROQ_API_KEY","GOOGLE_API_KEY"]
        self.api_keys = {key : os.getenv(key) for key in required_varibale}
        missing = [k for k,v in self.api_keys.items() if not v]
        if missing:
            log.error(f"Missing required environment variables: {missing}",missing_var = missing)
            raise AutoML_Exception(f"Missing required environment variables: {missing}",sys)

        log.info("Environment variables validated successfully", available_keys = list(self.api_keys.keys())) 

    def load_embeddings(self):
        try:
            log.info("Loading embeddings")
            model_name = self.config["embedding_model"]["model_name"]
            return GoogleGenerativeAIEmbeddings(model=model_name)
        except Exception as e:
            log.error(f"Error loading embeddings:",error = str(e))
            raise AutoML_Exception(f"Error loading embeddings: {e}", sys)

    def load_llm(self):
        llm_block = self.config["llm"]
        log.info("Loading LLM")

        provider_key = os.getenv("LLM_PROVIDER","groq")

        if provider_key not in llm_block:
            log.error(f"LLM provider {provider_key} not found in config",provider_key=provider_key)
            raise AutoML_Exception(f"Provider {provider_key} not found in LLM configuration", sys)

        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_tokens", 2048)

        log.info("Loading LLM", provider=provider, model=model_name, temperature=temperature, max_tokens=max_tokens)

        if provider == "google":
            llm = ChatGoogleGenerativeAI(
                model = model_name,
                temperature = temperature,
                max_output_tokens = max_tokens
            )
            return llm 
        
        elif provider == "groq":
            llm = ChatGroq(
                model = model_name,
                api_key = self.api_keys["GROQ_API_KEY"],
                temperature = temperature
            )
            return llm
        
        else :
            log.error("Unsupport LLM Provider",provider = provider)
            raise ValueError(f"Unsupport LLM Provider: {provider}")

if __name__ == "__main__":
    ml = ModelLoader() 
    
    llm = ml.load_llm()
    print(f"LLM loaded: {llm}")
    
    result = llm.invoke("Hello, how are you?")
    print(f"LLM response: {result.content}")