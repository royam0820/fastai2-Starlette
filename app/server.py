import aiohttp
import asyncio
import uvicorn
from fastai2.vision.all import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

import json
#from fastai2_inference import Inferencer


# export_file_url = YOUR_GDRIVE_LINK_HERE
export_file_url = 'https://www.googleapis.com/drive/v3/files/1RKuRyjlLi4A7JNiZ9xZN1-U_WPXxMaOs?alt=media&key=AIzaSyDwW_lF9ZU9gguzewL_DocSKcmMEjk_CGc'
export_file_name = 'export.pkl'

# classes = YOUR_CLASSES_HERE
classes = ['black', 'grizzly','teddy']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def get_bytes(url):
  async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
      return await response.read()

async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = torch.load(path/export_file_name, map_location=torch.device('cpu'))
        learn.dls.device = 'cpu'
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

            

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
  img_data = await request.form()
  img_bytes = await (img_data['file'].read())
  pred = learn.predict(BytesIO(img_bytes))[0]
  return JSONResponse({
      'results': str(pred)
  })


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
