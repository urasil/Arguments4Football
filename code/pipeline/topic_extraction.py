import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai
import time

class ExtractTopic:
    
    def __init__(self):
        
        load_dotenv()
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(
            model_name = "gemini-1.5-flash",
            system_instruction = """
                        You are a topic identification model that determines the topic of a group of arguments. The topic you determine should not team, player, manager etc. specific. You should be abstract in
                        the topic you have identified.

                        For example:

                        Everton's lack of consistency in performance, especially in attack and defense, shows that they struggle to maintain competitive results, leading to their underperformance.
                        Southampton's ineffective transfer policy, failing to reinvest properly in the squad, has resulted in an unbalanced team that cannot compete consistently in the Premier League.
                        Watford's managerial instability, with frequent changes in leadership, has prevented the team from developing a cohesive tactical approach, contributing to their underachievement.
                        Leeds United's defensive vulnerabilities, arising from their high-pressing style, have left them exposed and contributed to their underperformance in the league.
                        Newcastle United's persistent injuries to key players, like Callum Wilson and Allan Saint-Maximin, undermine their ability to compete effectively, resulting in a disappointing season.
                        Aston Villa's poor squad depth, which leaves them vulnerable when key players are unavailable, directly contributes to their inconsistent performances and underachievement.
                        Crystal Palace's lack of effective leadership, particularly when Wilfried Zaha is not performing, causes the team to struggle in crucial moments and underperform in the Premier League.
                        
                        The above 7 sentences all mention specific teams underperforming. Therefore the topic for this group of sentences can be "Underperforming Temams". Make sure your determined topic is brief and
                        concise. The format of your response should be the following:
                        Determined Topic: [The topic you determined],
                        [Explanation of determined topic]
                        """
        )

    def test_gemini(self):
        response = self.model.generate_content("Write a story about a magic backpack.")
        print(response.text)

    def determine_topic(self, group):  
        group_text = "\n".join(group) 
        response = self.model.generate_content(f"""Determine the topic of the following group of sentences:
                                                {group_text}""")
        try:
            topic = response.text.split("Determined Topic:")[1].split(",")[0].strip()  # Extract the topic
        except IndexError:
            topic = "Unknown"
        
        return topic