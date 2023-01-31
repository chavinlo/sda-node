import json
import requests
import discord
import base64
import io
from discord.ext import commands
import re
import argparse

intents = discord.Intents.all()
bot = commands.Bot(command_prefix='!', intents=intents)

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--token', required=True, help="Token for the discord bot")
parser.add_argument('-p', '--port', default="5000", help="localhost port for API access")
args = parser.parse_args()

@bot.command()
async def gen(ctx, *, args=None):
    if args == "help":
        await ctx.send('Parameters: !gen prompt="X" negprompt="X" steps="Multiple of 5, minimun 25" width="Multiple of 8, maximun 1024" height="Multiple of 8, maximun 1024" cfg="X" seed="X" and scheduler="EULER-A"/"DPMS"/"LMSD"')
        return

    print(args)

    match = re.findall(r"(\w+)=(?:\"|')([\S\s]+?)(?:\"|')", args)
    print(match)

    # create variables with the matched keys and values
    data = {}
    for key, value in match:
        data[key] = value

    print(data)
    # set default values for variables if not provided
    prompt = data['prompt'] if 'prompt' in data else None
    negprompt = data['negprompt'] if 'negprompt' in data else 'bad anatomy, (more than two arm per body:1.5), (more than two leg per body:1.5), (more than five fingers on one hand:1.5), multi arms, multi legs, bad arm anatomy, bad leg anatomy, bad hand anatomy, bad finger anatomy, bad detailed background, unclear architectural outline,lowres, worst quality'
    width = int(data['width']) if 'width' in data else 512
    height = int(data['height']) if 'height' in data else 512
    steps = int(data['steps']) if 'steps' in data else 25
    cfg = data['cfg'] if 'cfg' in data else 8
    seed = data['seed'] if 'seed' in data else -1
    scheduler = data['scheduler'] if 'scheduler' in data else "EULER-A"

    if prompt is None:
        await ctx.send("Error: A prompt must be provided")
        return
    if steps > 70:
        await ctx.send("Error: Max step count is 70")
        return
    if height > 1024 or width > 1024:
        await ctx.send("Error: Max resolution (width or height) is 1024")
        return
    if "0-mod" not in [y.name.lower() for y in ctx.message.author.roles]:
        await ctx.send("This command can only be used in the designated channel")
        return
    await ctx.send("Processing...")
    data = {
        "prompt": prompt,
        "negprompt": negprompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
        "scheduler": scheduler,
        "lpw": True,
        "mode": "json"
    }
    print("sending request:", data)

    response = requests.post(f"http://127.0.0.1:{args.port}/base", json=data)

    if response.status_code is not 200:
        ctx.send(response.text)

    response_data = json.loads(response.text)

    if response_data["status"] == "done":
        img_b64 = response_data["content"]["img"]
        latency = response_data["content"]["time"]
        seed = response_data["content"]["seed"]
        img = base64.b64decode(img_b64)
        file = discord.File(io.BytesIO(img), "image.png")
        embed = discord.Embed()
        embed.set_image(url="attachment://image.png")
        embed.set_footer(text=f"Latency: {round(latency, 3)}s | Seed: {seed}")
        await ctx.send(file=file, embed=embed)
    else:
        await ctx.send(response_data["content"])

bot.run(args.token)
