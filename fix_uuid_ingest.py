import re

with open("ingest.py", "r") as f:
    text = f.read()

# The error in ingest.py is that it calls uuid.uuid4().hex multiple times resulting in different UUIDs
# OR wait, no, the error is probably that multiple charts had the SAME heading and the uuid wasn't creating a unique ID?
# Let's check how chart_id is generated: chart_id = f"{filename}_chart_{uuid.uuid4().hex}"
# This should be unique every time. 

# WAIT. Is Chromadb collection.add overwriting because it's called in a loop? No, collection.add with a unique ID appends.
# Let's look at the output of python ingest.py
# "Found chart under heading 'Alphabet Revenues and Operating Income', fetching Gemini description..."
# But wait, did it only find ONE chart?
# No, "PICTURE FOUND" happened 9 times.
# Ah, Docling `element.label.name.startswith('heading')` might not be updating `current_heading` correctly, 
# But that wouldn't stop it from inserting. 

# Let's check `process_document` in ingest.py
