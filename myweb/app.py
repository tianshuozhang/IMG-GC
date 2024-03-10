# -*- coding: utf-8 -*-
from enum import Enum
from flask import Flask, render_template, request, flash, redirect, url_for
from markupsafe import Markup
from flask_wtf import FlaskForm, CSRFProtect
from wtforms.validators import DataRequired, Length, Regexp
from wtforms.fields import *
from flask_bootstrap import Bootstrap5, SwitchField
from flask_sqlalchemy import SQLAlchemy

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
db = SQLAlchemy(app)
csrf = CSRFProtect(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/img2img',methods=['GET', 'POST'])
def img2img():
    form =Img2imgForm()
    if form.validate_on_submit():
        return "123"
    return render_template('img2img.html',form=form)

@app.route('/img2text',methods=['GET', 'POST'])
def img2text():
    form = Img2textForm()
    if form.validate_on_submit():
        return "123"
    return render_template('img2text.html',form=form)

class Img2imgForm(FlaskForm):
    url = URLField()
    image = FileField(render_kw={'class': 'my-class'})  # add your class
    select = SelectField(choices=[('url_load', 'url_load'), ('image_load', 'image_load')])
    Prompt = TextAreaField(validators=[DataRequired()])
    submit = SubmitField()

class Img2textForm(FlaskForm):
    url = URLField()
    image = FileField(render_kw={'class': 'my-class'})  # add your class
    select = SelectField(choices=[('url_load', 'url_load'), ('image_load', 'image_load')])
    Prompt = TextAreaField()
    Question = TextAreaField()
    choice = SelectField(choices=[('question', 'question'), ('prompt', 'prompt')])
    submit = SubmitField()
