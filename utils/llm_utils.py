from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict, Any, Optional
import os

class LLMProcessor:
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        """
        Initialize LLM processor
        
        Args:
            model_name: Model name to use
            temperature: Parameter for generation diversity
        """
        self.model_name = model_name
        self.temperature = temperature
        self.chat = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        # Default system prompt
        self.default_system_prompt = """
        You are a professional customer service representative, capable of accurately understanding user needs and providing assistance.
        Your responses should be concise, friendly, and helpful.
        If you are unsure about an answer, please honestly state that you don't know rather than providing potentially inaccurate information.
        """
    
    def generate_response(self, prompt: str, 
                         conversation_history: Optional[List[Dict[str, str]]] = None, 
                         system_prompt: Optional[str] = None) -> str:
        """
        Generate response
        
        Args:
            prompt: User input
            conversation_history: Conversation history
            system_prompt: System prompt
            
        Returns:
            Generated response
        """
        # Use default system prompt or custom prompt
        system_content = system_prompt if system_prompt else self.default_system_prompt
        
        # Build message list
        messages = [SystemMessage(content=system_content)]
        
        # Add conversation history
        if conversation_history:
            for message in conversation_history:
                if message["role"] == "user":
                    messages.append(HumanMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    messages.append(AIMessage(content=message["content"]))
        
        # Add current user input
        messages.append(HumanMessage(content=prompt))
        
        # Generate response
        response = self.chat(messages)
        return response.content
    
    def customize_for_call_center(self) -> None:
        """
        Customize LLM for call center scenarios
        """
        self.default_system_prompt = """
        You are a professional call center customer service representative, capable of handling various customer inquiries and issues.
        
        Please follow these guidelines:
        1. Maintain a professional, friendly, and polite attitude
        2. Provide clear and concise answers, avoiding lengthy explanations
        3. Proactively offer relevant information, but don't oversell
        4. If you need more information to answer a question, politely ask for it
        5. If you cannot resolve the customer's issue, offer the option to escalate to a human agent
        
        Remember, your goal is to efficiently resolve customer issues while providing a good customer experience.
        """
    
    def customize_for_lead_generation(self) -> None:
        """
        Customize LLM for lead generation scenarios
        """
        self.default_system_prompt = """
        You are a professional sales representative, responsible for initial communication with potential customers and collecting information.
        
        Please follow these guidelines:
        1. Introduce yourself and your company in a friendly manner
        2. Inquire about potential customers' needs and pain points
        3. Briefly introduce how relevant products or services can solve their problems
        4. Collect key information (such as contact details, best time to contact, etc.)
        5. Suggest next steps (such as arranging a demonstration, sending materials, etc.)
        
        Remember, your goal is to establish an initial relationship and collect sufficient information for follow-up, not to complete the sale in the first conversation.
        """
    
    def analyze_conversation(self, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze conversation content and extract key information
        
        Args:
            conversation_history: Conversation history
            
        Returns:
            Analysis results containing key information
        """
        # Build analysis prompt
        analysis_prompt = """
        Please analyze the following conversation and extract the following information:
        1. Customer's main issues or needs
        2. Customer's emotional state
        3. Key information points (such as product interest, budget considerations, etc.)
        4. Suggested follow-up actions
        
        Conversation content:
        """
        
        # Add conversation history to prompt
        for message in conversation_history:
            role = "Customer" if message["role"] == "user" else "Assistant"
            analysis_prompt += f"\n{role}: {message['content']}"
        
        # Build messages
        messages = [
            SystemMessage(content="You are a professional conversation analysis expert, capable of extracting key information from conversations."),
            HumanMessage(content=analysis_prompt)
        ]
        
        # Generate analysis
        response = self.chat(messages)
        
        # More structured processing logic can be added here
        # Currently simply returns text analysis results
        return {"analysis": response.content}
    