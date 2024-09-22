from text_to_speech import save

text = "The river level has dropped to around 15 ft a few times this year, but it doesn’t stay there for long. It will continue to fluctuate up and down. Once the river drops to around 14 ft or lower the beach will have much more sand exposed, and it will be possible to walk on the river bed if you don’t mind getting a little wet."
language = "en"
output_file = "slow_speech.mp3"

save(text, language, slow=True, file=output_file)