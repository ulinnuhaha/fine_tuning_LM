from together import Together
import os

class FTModel:
    def __init__(self, llm_name):
        self.llm_name = llm_name
        self.api_key = os.getenv("TAI_API_KEY")  # Ensure API key is retrieved
        self.model_name = self.get_model_name()
        
        if self.model_name is None:
            raise ValueError("The input model is not available.")

    def get_model_name(self):
        """Retrieve the correct model name based on user input"""
        model_mapping = {
            'ft_llama_31_8b': "your_org/Meta-Llama-3.1-8B-Instruct-Reference-mt",
            'ft_llama_31_70b': "your_org/Meta-Llama-3.1-70B-Instruct-Reference-mt"
        }
        return model_mapping.get(self.llm_name)

    def generating(self, prompt_1, prompt_2, requested_translation):
        """Generate output using the Together API"""
        data_input = [
            {"role": "system", "content": prompt_1},
            {"role": "user", "content": prompt_2 + str(requested_translation)}
        ]

        if not self.api_key:
            raise ValueError("API key is missing. Please set the TAI_API_KEY environment variable.")

        client = Together(api_key=self.api_key)  # Use the actual API key

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=data_input,
                temperature=0.7,  # Adjust as needed
                max_tokens=512  # Adjust as needed
            )
            return response
        except Exception as e:
            print(f"Error during API call: {e}")
            return None
