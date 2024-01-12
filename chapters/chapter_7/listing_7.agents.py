from langchain.tools import DuckDuckGoSearchRun, YouTubeSearchTool

# Example of using tools
search = DuckDuckGoSearchRun()
hot_topic = search.run(
    "Tiktoker finds proof of Fruit of the Loom cornucopia in the logo"
)

youtube_tool = YouTubeSearchTool()
fun_channel = youtube_tool.run("jaubrey", 3)

print(hot_topic, fun_channel)
# Rating: False About this rating If asked to describe underwear
# manufacturer Fruit of the Loom's logo from memory, some will invariably
# say it includes — or at least included at some point in... A viral claim
# recently surfaced stating that Fruit of the Loom, the American underwear
# and casualwear brand, had a cornucopia in their logo at some point in the
# past. It refers to a goat's... The Fruit of the Loom Mandela Effect is
# really messing with people's memories of the clothing company's iconic
# logo.. A viral TikTok has thousands of people not only thinking about what
# they remember the logo to look like, but also has many searching for proof
# that we're not all losing our minds.. A TikTok Creator Is Trying To Get To
# The Bottom Of The Fruit Of The Loom Mandela Effect What Is 'The Mandela
# Effect?' To understand why people care so much about the Fruit of the Loom
# logo, one must first understand what the Mandela Effect is in the first
# place. It's a slang term for a cultural phenomenon in which a large group
# of people shares false memories of past events. About Fruit of the Loom
# Cornucopia and Fruit of the Loom Mandela Effect refer to the Mandela
# Effect involving a large number of people remembering the clothing company
# Fruit of the Loom having a cornucopia on its logo despite the logo never
# having the item on it.
# ['https://www.youtube.com/watch?v=x81gguSPGcQ&pp=ygUHamF1YnJleQ%3D%3D',
#'https://www.youtube.com/watch?v=bEvxuG6mevQ&pp=ygUHamF1YnJleQ%3D%3D']
