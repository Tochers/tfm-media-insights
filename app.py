# Libraries
import gradio as gr
import whisper
from pytube import YouTube
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from wordcloud import WordCloud
import re
import os


class GradioInference:
    def __init__(self):
        
        # OpenAI's Whisper model sizes
        self.sizes = list(whisper._MODELS.keys())

        # Whisper's available languages for ASR
        self.langs = ["none"] + sorted(list(whisper.tokenizer.LANGUAGES.values()))
        
        # Default size
        self.current_size = "base"
        
        # Default model size
        self.loaded_model = whisper.load_model(self.current_size)
        
        # Initialize Pytube Object
        self.yt = None

        # Initialize summary model for English
        self.bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn", truncation=True)

        # Initialize Multilingual summary model 
        self.mt5_tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum", truncation=True)
        self.mt5_model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")

        # Initialize VoiceLabT5 model and tokenizer
        self.keyword_model = T5ForConditionalGeneration.from_pretrained(
            "Voicelab/vlt5-base-keywords"
        )
        self.keyword_tokenizer = T5Tokenizer.from_pretrained(
            "Voicelab/vlt5-base-keywords"
        )

        # Sentiment Classifier
        self.classifier = pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", return_all_scores=False)

    
    def __call__(self, link, lang, size, progress=gr.Progress()):
        """
        Call the Gradio Inference python class.
        This class gets access to a YouTube video using python's library Pytube and downloads its audio.
        Then it uses the Whisper model to perform Automatic Speech Recognition (i.e Speech-to-Text).
        Once the function has the transcription of the video it proccess it to obtain:
            - Summary: using Facebook's BART transformer.
            - KeyWords: using VoiceLabT5 keyword extractor.
            - Sentiment Analysis: using Hugging Face's default sentiment classifier
            - WordCloud: using the wordcloud python library.
        """
        try:
            
            progress(0, desc="Starting analysis")
            
            if self.yt is None:
                self.yt = YouTube(link)
            
            # Pytube library to access to YouTube audio stream
            path = self.yt.streams.filter(only_audio=True)[0].download(filename="tmp.mp4")
    
            if lang == "none":
                lang = None
    
            if size != self.current_size:
                self.loaded_model = whisper.load_model(size)
                self.current_size = size
    
            progress(0.20, desc="Transcribing")
            
            # Transcribe the audio extracted from pytube
            results = self.loaded_model.transcribe(path, language=lang)
    
            progress(0.40, desc="Summarizing")
            
            # Perform summarization on the transcription
            transcription_summary = self.bart_summarizer(
                results["text"], 
                max_length=150, 
                min_length=30, 
                do_sample=False, 
                truncation=True
            )
    
            # Multilingual summary with mt5 
            WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
            
            input_ids_sum = self.mt5_tokenizer(
                [WHITESPACE_HANDLER(results["text"])],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )["input_ids"]
            
            output_ids_sum = self.mt5_model.generate(
                input_ids=input_ids_sum,
                max_length=256,
                no_repeat_ngram_size=2,
                num_beams=4
            )[0]
            
            summary = self.mt5_tokenizer.decode(
                output_ids_sum,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            # End multilingual summary
            
            progress(0.60, desc="Extracting Keywords")
            
            # Extract keywords using VoiceLabT5
            task_prefix = "Keywords: "
            input_sequence = task_prefix + results["text"]
            
            input_ids = self.keyword_tokenizer(
                input_sequence, 
                return_tensors="pt", 
                truncation=False
            ).input_ids
            
            output = self.keyword_model.generate(
                input_ids, 
                no_repeat_ngram_size=3, 
                num_beams=4
            )
            
            predicted = self.keyword_tokenizer.decode(output[0], skip_special_tokens=True)
            keywords = [x.strip() for x in predicted.split(",") if x.strip()]
            formatted_keywords = "\n".join([f"• {keyword}" for keyword in keywords])
    
            progress(0.80, desc="Extracting Sentiment")
           
            # Define a dictionary to map labels to emojis
            sentiment_emojis = {
                "positive": "Positive 👍🏼",
                "negative": "Negative 👎🏼",
                "neutral": "Neutral 😶",
            }
            
            # Sentiment label    
            label = self.classifier(summary)[0]["label"]
    
            # Format the label with emojis
            formatted_sentiment = sentiment_emojis.get(label, label)
    
            progress(0.90, desc="Generating Wordcloud")
            
            # Generate WordCloud object
            wordcloud = WordCloud(colormap = "Oranges").generate(results["text"])
    
            # WordCloud image to display
            wordcloud_image = wordcloud.to_image()
    
            if lang == "english" or lang == "none":
                return (
                    results["text"],
                    transcription_summary[0]["summary_text"],
                    formatted_keywords,
                    formatted_sentiment,
                    wordcloud_image,
                )
            else:
                return (
                    results["text"],
                    summary,
                    formatted_keywords,
                    formatted_sentiment,
                    wordcloud_image,
                )
            
        except:
            gr.Error(message="Restricted Content. Choose a different video")

        finally:
            gr.Info("Success!")
    

    def populate_metadata(self, link):
        """
        Access to the YouTube video title and thumbnail image to further display it
        params:
        - link: a YouTube URL.
        """
        if not link:
            return None, None
            
        self.yt = YouTube(link)
        return self.yt.thumbnail_url, self.yt.title

    def from_audio_input(self, lang, size, audio_file, progress=gr.Progress()):
        """
        Call the Gradio Inference python class.
        Uses it directly the Whisper model to perform Automatic Speech Recognition (i.e Speech-to-Text).
        Once the function has the transcription of the video it proccess it to obtain:
            - Summary: using Facebook's BART transformer.
            - KeyWords: using VoiceLabT5 keyword extractor.
            - Sentiment Analysis: using Hugging Face's default sentiment classifier
            - WordCloud: using the wordcloud python library.
        """
        try:
            progress(0, desc="Starting analysis")
            
            if lang == "none":
                lang = None
    
            if size != self.current_size:
                self.loaded_model = whisper.load_model(size)
                self.current_size = size
    
            progress(0.20, desc="Transcribing")
            
            results = self.loaded_model.transcribe(audio_file, language=lang)
    
            progress(0.40, desc="Summarizing")
            
            # Perform summarization on the transcription
            transcription_summary = self.bart_summarizer(
                results["text"], max_length=150, min_length=30, do_sample=False, truncation=True
            )
            
            # Multilingual summary with mt5
            WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
            
            input_ids_sum = self.mt5_tokenizer(
                [WHITESPACE_HANDLER(results["text"])],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )["input_ids"]
            
            output_ids_sum = self.mt5_model.generate(
                input_ids=input_ids_sum,
                max_length=130,
                no_repeat_ngram_size=2,
                num_beams=4
            )[0]
            
            summary = self.mt5_tokenizer.decode(
                output_ids_sum,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            # End multilingual summary
    
            progress(0.60, desc="Extracting Keywords")
            
            # Extract keywords using VoiceLabT5
            task_prefix = "Keywords: "
            input_sequence = task_prefix + results["text"]
            
            input_ids = self.keyword_tokenizer(
                input_sequence, 
                return_tensors="pt", 
                truncation=False
            ).input_ids
            
            output = self.keyword_model.generate(
                input_ids, 
                no_repeat_ngram_size=3, 
                num_beams=4
            )
            predicted = self.keyword_tokenizer.decode(output[0], skip_special_tokens=True)
            keywords = [x.strip() for x in predicted.split(",") if x.strip()]
            formatted_keywords = "\n".join([f"• {keyword}" for keyword in keywords])
    
            progress(0.80, desc="Extracting Sentiment")
    
            # Define a dictionary to map labels to emojis
            sentiment_emojis = {
                "positive": "Positive 👍🏼",
                "negative": "Negative 👎🏼",
                "neutral": "Neutral 😶",
            }
            
            # Sentiment label    
            label = self.classifier(summary)[0]["label"]
    
            # Format the label with emojis
            formatted_sentiment = sentiment_emojis.get(label, label)
            
            progress(0.90, desc="Generating Wordcloud")
            # WordCloud object
            wordcloud = WordCloud(colormap = "Oranges").generate(
                results["text"]
            )
            wordcloud_image = wordcloud.to_image()
    
            if lang == "english" or lang == "none":
                return (
                    results["text"],
                    transcription_summary[0]["summary_text"],
                    formatted_keywords,
                    formatted_sentiment,
                    wordcloud_image,
                )
            else:
                return (
                    results["text"],
                    summary,
                    formatted_keywords,
                    formatted_sentiment,
                    wordcloud_image,
                )

        except:
            gr.Error(message="Exceeded audio size. Choose a different audio")

        finally:
            gr.Info("Success!")
    
    def from_article(self, article, progress=gr.Progress()):
        """
        Call the Gradio Inference python class.
        Acepts the user's text imput, then it performs: 
            - Summary: using Facebook's BART transformer.
            - KeyWords: using VoiceLabT5 keyword extractor.
            - Sentiment Analysis: using Hugging Face's default sentiment classifier
            - WordCloud: using the wordcloud python library.
        """
        try:
            progress(0, desc="Starting analysis")
    
            progress(0.30, desc="Summarizing")
            
            # Perform summarization on the transcription
            transcription_summary = self.bart_summarizer(
                article, max_length=150, min_length=30, do_sample=False, truncation=True
            )
            
            # Multilingual summary with mt5
            WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
            
            input_ids_sum = self.mt5_tokenizer(
                [WHITESPACE_HANDLER(article)],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )["input_ids"]
            
            output_ids_sum = self.mt5_model.generate(
                input_ids=input_ids_sum,
                max_length=130,
                no_repeat_ngram_size=2,
                num_beams=4
            )[0]
            
            summary = self.mt5_tokenizer.decode(
                output_ids_sum,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            # End multilingual summary
    
            progress(0.60, desc="Extracting Keywords")
            
            # Extract keywords using VoiceLabT5
            task_prefix = "Keywords: "
            input_sequence = task_prefix + article
            
            input_ids = self.keyword_tokenizer(
                input_sequence, 
                return_tensors="pt", 
                truncation=False
            ).input_ids
            
            output = self.keyword_model.generate(
                input_ids, 
                no_repeat_ngram_size=3, 
                num_beams=4
            )
            predicted = self.keyword_tokenizer.decode(output[0], skip_special_tokens=True)
            keywords = [x.strip() for x in predicted.split(",") if x.strip()]
            formatted_keywords = "\n".join([f"• {keyword}" for keyword in keywords])
    
            progress(0.80, desc="Extracting Sentiment")
    
            # Define a dictionary to map labels to emojis
            sentiment_emojis = {
                "positive": "Positive 👍🏼",
                "negative": "Negative 👎🏼",
                "neutral": "Neutral 😶",
            }
            
            # Sentiment label    
            label = self.classifier(summary)[0]["label"]
    
            # Format the label with emojis
            formatted_sentiment = sentiment_emojis.get(label, label)
            
            progress(0.90, desc="Generating Wordcloud")
            # WordCloud object
            wordcloud = WordCloud(colormap = "Oranges").generate(
                article
            )
            wordcloud_image = wordcloud.to_image()
    
            return (
                transcription_summary[0]["summary_text"],
                formatted_keywords,
                formatted_sentiment,
                wordcloud_image,
            )

        except:
            gr.Error(message="Exceeded text size. Choose a different audio")

        finally:
            gr.Info("Success!")

gio = GradioInference()
title = "Media Insights"
description = "Your AI-powered video analytics tool"
theme = gr.themes.Soft(spacing_size="md", radius_size="md")

block = gr.Blocks(theme=theme)

# Gradio Interface
with block as demo:
    # Title
    gr.HTML(
        """
        <div style="text-align: center; max-width: 500px; margin: 0 auto;">
          <div>
            <h1 style="font-family: Poppins, sans-serif;">MEDIA <span style="color: #433ccb;">INSIGHTS</span> 💡</h1>
          </div>
          <h4>
            Your AI-powered media analytics tool ✨
          </h4>
        </div>
        """
    )
    # Group of tabs
    with gr.Group():
        with gr.Tab("From YouTube 📹"):
            with gr.Box():

                # Model Size and Language selections
                with gr.Row().style(equal_height=True):
                    size = gr.Dropdown(
                        label="Speech-to-text Model Size", choices=gio.sizes, value="base"
                    )
                    lang = gr.Dropdown(
                        label="Language (Optional)", choices=gio.langs, value="none"
                    )
                link = gr.Textbox(
                    label="YouTube Link", placeholder="Enter YouTube link..."
                )

                # Video Metadata
                with gr.Row().style(equal_height=True):
                    with gr.Column(variant="panel", scale=1):
                        title = gr.Label(label="Video Title")
                        img = gr.Image(label="Thumbnail").style(height=350)

                    # Video Transcription
                    with gr.Column(variant="panel", scale=1):
                        text = gr.Textbox(
                            label="Transcription",
                            placeholder="Transcription Output...",
                            lines=18,
                        ).style(show_copy_button=True)

                # Video block of summary, keywords , sent. analysis and wordcloud
                with gr.Row().style(equal_height=True):
                    summary = gr.Textbox(
                        label="Summary", placeholder="Summary Output...", lines=5
                    ).style(show_copy_button=True)
                    keywords = gr.Textbox(
                        label="Keywords", placeholder="Keywords Output...", lines=5
                    ).style(show_copy_button=True)
                    label = gr.Label(label="Sentiment Analysis")
                    wordcloud_image = gr.Image(label="WordCloud")

            # Buttons
            with gr.Row():
                btn = gr.Button("Get Video Insights 🔎", variant="primary", scale=1)
                clear = gr.ClearButton(
                        [link, title, img, text, summary, keywords, label, wordcloud_image],
                        value="Clear 🗑️", scale=1
                )
            btn.click(
                gio,
                inputs=[link, lang, size],
                outputs=[text, summary, keywords, label, wordcloud_image],
            )
            link.change(gio.populate_metadata, inputs=[link], outputs=[img, title])

        with gr.Tab("From Audio file 🎙️"):
            with gr.Box():

                # Model selections
                with gr.Row().style(equal_height=True):
                    size = gr.Dropdown(
                        label="Model Size", choices=gio.sizes, value="base"
                    )
                    lang = gr.Dropdown(
                        label="Language (Optional)", choices=gio.langs, value="none"
                    )
                audio_file = gr.Audio(type="filepath")

                # Audio transcription
                with gr.Row().style(equal_height=True):
                    text = gr.Textbox(
                        label="Transcription",
                        placeholder="Transcription Output...",
                        lines=10,
                    ).style(show_copy_button=True)

                # Audio analysis
                with gr.Row().style(equal_height=True):
                    summary = gr.Textbox(
                        label="Summary", placeholder="Summary Output...", lines=5
                    )
                    keywords = gr.Textbox(
                        label="Keywords", placeholder="Keywords Output...", lines=5
                    )
                    label = gr.Label(label="Sentiment Analysis")
                    wordcloud_image = gr.Image(label="WordCloud")
                    
            with gr.Row():
                btn = gr.Button(
                    "Get Audio Insights 🔎", variant="primary"
                )
                clear = gr.ClearButton([audio_file,text, summary, keywords, label, wordcloud_image], value="Clear 🗑️")
            btn.click(
                gio.from_audio_input,
                inputs=[lang, size, audio_file],
                outputs=[text, summary, keywords, label, wordcloud_image],
            )

        with gr.Tab("From Article 📋"):
            with gr.Box():

                # Text input from user
                with gr.Row().style(equal_height=True):
                    article = gr.Textbox(
                        label="Text",
                        placeholder="Paste your text...",
                        lines=10,
                    ).style(show_copy_button=True)

                # Text analysis
                with gr.Row().style(equal_height=True):
                    summary = gr.Textbox(
                        label="Summary", placeholder="Summary Output...", lines=5
                    )
                    keywords = gr.Textbox(
                        label="Keywords", placeholder="Keywords Output...", lines=5
                    )
                    label = gr.Label(label="Sentiment Analysis")
                    wordcloud_image = gr.Image(label="WordCloud")
                    
            with gr.Row(): 
                btn = gr.Button(
                    "Get Text insights 🔎", variant="primary")
                clear = gr.ClearButton([article, summary, keywords, label, wordcloud_image], value="Clear")
            btn.click(
                gio.from_article,
                inputs=[article],
                outputs=[summary, keywords, label, wordcloud_image],
            )

# Open text example
with open(os.path.join(os.path.dirname(__file__), "texts/India_Canada.txt"), "r") as file:
    text_example_content = file.read()


with block:
    # Video Examples
    gr.Markdown("### Video Examples")
    gr.Examples(["https://www.youtube.com/shorts/xDNzz8yAH7I",
                 "https://youtu.be/MnrJzXM7a6o",
                "https://youtu.be/FKjj1tNcbtM"],
                inputs=link)

    # Audio Examples
    gr.Markdown("### Audio Examples")
    gr.Examples([[os.path.join(os.path.dirname(__file__),"audios/EnglishLecture.mp4")]], inputs=audio_file)

    # Text Examples
    gr.Markdown("### Text Examples")
    with gr.Accordion("News text example", open=False):
        gr.Examples([[text_example_content]], inputs=article)

    # FAQs section
    gr.Markdown("### About the app:")
    
    with gr.Accordion("What is Media Insights?", open=False):
        gr.Markdown(
            "Media Insights is a tool developed for academic purposes that allows you to analyze YouTube videos, audio files or some text. It provides features like transcription, summarization, keyword extraction, sentiment analysis, and word cloud generation for multimedia content."
        )

    with gr.Accordion("How does Media Insights work?", open=False):
        gr.Markdown(
            "Media Insights leverages several powerful AI models and libraries. It uses OpenAI's Whisper for Automatic Speech Recognition (ASR) to transcribe audio content. It summarizes the transcribed text using Facebook's BART model, extracts keywords with VoiceLabT5, performs sentiment analysis with DistilBERT, and generates word clouds."
        )

    with gr.Accordion("What languages are supported for the analysis?", open=False):
        gr.Markdown(
            "Media Insights supports multiple languages for transcription and analysis. You can select your preferred language from the available options when using the app."
        )

    with gr.Accordion("Can I analyze audio files instead of YouTube videos?", open=False):
        gr.Markdown(
            "Yes, you can analyze audio files directly. Simply upload your audio file to the app, and it will provide the same transcription, summarization, keyword extraction, sentiment analysis, and word cloud generation features. In addition, you can also paste your article or text of your preference, to get all the insights directly from it."
        )

    with gr.Accordion("What are the different model sizes available for transcription?", open=False):
        gr.Markdown(
            "The app uses a Speech-to-text model that has different training sizes, from tiny to large. Hence, the bigger the model the accurate the transcription."
        )

    with gr.Accordion("How long does it take to analyze a video or audio file?", open=False):
        gr.Markdown(
            "The time taken for analysis may vary based on the duration of the video or audio file and the selected model size. Shorter content will be processed more quickly."
        )
    
    with gr.Accordion("Who developed Media Insights?" ,open=False):
        gr.Markdown(
            "Media Insights was developed by students as part of the 2022/23 Master's in Big Data & Data Science program at Universidad Complutense de Madrid for academic purposes (Trabajo de Fin de Master)."
        )

    # Page footer
    gr.HTML(
        """
        <div style="text-align: center; margin: 0 auto;">
          <p style="margin-bottom: 10px; font-size: 96%">
            Trabajo de Fin de Máster - Grupo 3
          </p>
          <p>
            2022/23 Master in Big Data & Data Science - Universidad Complutense de Madrid
          </p>
        </div>
        """
    )


demo.launch()

