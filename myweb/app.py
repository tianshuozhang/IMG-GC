# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, flash, redirect, url_for
from flask_wtf import FlaskForm, CSRFProtect
from wtforms.validators import DataRequired, Length, Regexp
from wtforms.fields import *
from flask_bootstrap import Bootstrap5, SwitchField
from utils.get_input import *
from utils.models import *
from diffusers import StableDiffusionImg2ImgPipeline
from wtforms.validators import Optional
import shutil
import os
import threading
app = Flask(__name__)
app.secret_key = 'dev'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'

# set default button sytle and size, will be overwritten by macro parameters
app.config['BOOTSTRAP_BTN_STYLE'] = 'primary'
app.config['BOOTSTRAP_BTN_SIZE'] = 'sm'

# set default icon title of table actions
app.config['BOOTSTRAP_TABLE_VIEW_TITLE'] = 'Read'
app.config['BOOTSTRAP_TABLE_EDIT_TITLE'] = 'Update'
app.config['BOOTSTRAP_TABLE_DELETE_TITLE'] = 'Remove'
app.config['BOOTSTRAP_TABLE_NEW_TITLE'] = 'Create'

bootstrap = Bootstrap5(app)
csrf = CSRFProtect(app)
processor = None
model_bilp = None
pipe=None
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/img2img',methods=['GET', 'POST'])
def img2img():
    form =Img2imgForm()
    if form.validate_on_submit():
        select = form.select.data
        if select == 'url_load':
            local_img = False
            url = form.url.data
        else:
            local_img = True
            url = form.image.data.filename
            # 当前文件的绝对路径
            current_path = os.path.abspath(url)
            # 目标文件夹
            target_folder = os.path.abspath("./myweb/static/" + url)
            # 移动文件
            shutil.copy(current_path, target_folder)
        prompt=form.Prompt.data
        raw_image = get_img(url=url, local_img=local_img)
        image = pipe(image=raw_image, prompt=prompt, strength=0.75, guidance_scale=7.5).images[0]
        image.save("./myweb/static/{}.jpg".format(prompt))
        return render_template('img_show.html',image_name=url,select=select,prompt=prompt)
    thread = threading.Thread(target=load_pipe)
    thread.start()
    print("pipe start")
    return render_template('img2img.html',form=form)

@app.route('/img2text',methods=['GET', 'POST'])
def img2text():
    form = Img2textForm()
    if form.validate_on_submit():
        select = form.select.data
        choice = form.choice.data
        if select=='url_load':
            local_img=False
            url=form.url.data
        else:
            local_img = True
            url = form.image.data.filename

            # 当前文件的绝对路径
            current_path = os.path.abspath(url)
            # 目标文件夹
            target_folder = os.path.abspath("./myweb/static/" + url)
            # 移动文件
            shutil.copy(current_path, target_folder)

        if choice=='question':
            prompt = "Question:"+form.Question.data+"? Answer:"
        else:
            prompt = form.Prompt.data
        inputs = get_input(processor=processor, url=url, text=prompt, local_img=local_img)
        out = model_bilp.generate(**inputs, max_length=200)
        question = form.Question.data if choice=='question' else prompt
        answer = processor.decode(out[0], skip_special_tokens=True).strip()
        return render_template("text_show.html",image_name=url,select=select,choice=choice,answer=answer,question=question)
    thread = threading.Thread(target=load_model_bilp)
    thread.start()
    return render_template('img2text.html',form=form)

class Img2imgForm(FlaskForm):
    url = URLField(validators=[Optional()])
    image = FileField(render_kw={'class': 'my-class'})  # add your class
    select = SelectField(choices=[('url_load', 'url_load'), ('image_load', 'image_load')])
    Prompt = TextAreaField(validators=[DataRequired()])
    submit = SubmitField()

class Img2textForm(FlaskForm):
    url = URLField(validators=[Optional()])
    image = FileField(render_kw={'class': 'my-class'})  # add your class
    select = SelectField(choices=[('url_load', 'url_load'), ('image_load', 'image_load')])
    Prompt = TextAreaField()
    Question = TextAreaField()
    choice = SelectField(choices=[('question', 'question'), ('prompt', 'prompt')])
    submit = SubmitField()
def load_model_bilp():
    global processor,model_bilp
    if model_bilp is not None:
        return
    device_map = {'language_model': "cuda", \
                  'language_projection': 'cpu', \
                  'qformer': 'cpu', \
                  'query_tokens': 'cpu', \
                  'vision_model': 'cpu'}
    model_path = "Salesforce/blip2-opt-2.7b"
    processor , model_bilp = get_processor_and_model(modelpath=model_path, device_map=device_map)
    lora_fit(model_bilp)

def load_pipe():
    global pipe
    if pipe is not None:
        return
    model_path = "stabilityai/stable-diffusion-2"
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch.float16, revision="fp16", use_auth_token=True , device_map='sequential')
    # pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_path,  use_auth_token=True)
