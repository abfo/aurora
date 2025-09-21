# Aurora

Aurora is an Alexa / Google Home style assistant implemented in python. The project includes a headless UX
that just needs audio, and an implementation that runs on Raspberry PI 4 with the [Adafruit BrainCraft HAT](https://learn.adafruit.com/adafruit-braincraft-hat-easy-machine-learning-for-raspberry-pi).

Aurora uses the OpenAI realtime API. You need an [OpenAI API Key](https://platform.openai.com/). Wake word detection
uses [Picovoice Porcupine](https://picovoice.ai/platform/porcupine/) and you need a free API key and model file.

![Aurora](/assets/aurora.png)

Read the [release post](https://ithoughthecamewithyou.com/post/aurora-the-raspberry-pi-smart-assistant) on my blog. 

## Tools

Aurora supports a plugin system for tools. The following are included:

1. Timers - set, list and delete timers by name. 
2. Cooking - help cook a recipe step by step. 
2. Perplexity Sonar - look up information using the Sonar API.
3. Todoist - add shopping and to do list items. 
4. LIFX light control - turn on and off LIFX smart light bulbs.
5. Next Transit - get predicted arrival times for a configured Bay Area public transit route.
6. Cheese Night - a simple tool to decide which kid gets the first pick of chores.

Tools are only included in realtime calls if configured (see .env.example). To add a new tool just inherit from the Tool base class and add any required settings. If you add something useful please send a pull request. The realtime API has support for server MCP in addition to local tools, this is not supported yet but would be straightforward to add.

## Headless Configuration

Get the headless version up and running:

1. Clone this repo.
2. Create a Python virtual environment (recommended).
3. Install common dependencies (`pip install -r requirements.txt`).
4. Copy .env.example to .env and follow the instructions within to configure. 
5. Run main.py and start talking!

## Adafruit BrainCraft Configuration

This is the version pictured above. 

1. Follow the headless instructions above.
2. Install BrainCraft dependencies (`pip install -r requirements-braincraft.txt`)
3. In .env change `UI` to `Braincraft` and provide paths to required assets (the assets folder of this
project has some examples to get you started). 
4. Run main.py and start talking! 

Once you have a Raspberry PI 4 and BrainCraft HAT follow [these instructions](https://learn.adafruit.com/adafruit-braincraft-hat-easy-machine-learning-for-raspberry-pi/raspberry-pi-setup) to make sure the HAT 
is configured and working properly. 

A simple [case](/case/case.md) can be printed to house this version.

## Contributing 

Pull requests welcome. Aurora started as a side project to explore LLM based voice assistants. I expected a lot of commercial and open source alternatives but so far nothing has hit the market that I like as much as this implementation. I don't want to be marketed to by Alexa, but I also don't want a local-LLM privacy first assistant that sacrifices intelligence and flexibility. Aurora uses the state of the art and prioritizes power over cost. If you like this vision then help make this better - more tools and platforms especially welcome.


