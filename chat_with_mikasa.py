#!/usr/bin/env python3
import torch
import os
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import sys
import platform
import gradio as gr
from gradio import ChatMessage
from threading import Thread

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_and_tokenizer(config, model_path="./mikasa-ft"):
    print("Loading Mikasa... Please wait, senpai~")

    is_mac = platform.system() == "Darwin"

    model_kwargs = {
        "trust_remote_code": True,
    }

    if is_mac:
        print("Detected Mac - using MPS if available")
        if torch.backends.mps.is_available():
            model_kwargs["device_map"] = {"": "mps"}
            model_kwargs["dtype"] = torch.float16
        else:
            model_kwargs["device_map"] = {"": "cpu"}
            model_kwargs["dtype"] = torch.float32
    else:
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            model_kwargs["dtype"] = torch.float16
        else:
            model_kwargs["device_map"] = {"": "cpu"}
            model_kwargs["dtype"] = torch.float32

    # Load tokenizer first from fine-tuned directory to get the correct vocab size
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect whether mikasa-ft is a full model or an adapter-only folder
    has_full = any(
        os.path.exists(os.path.join(model_path, fname)) for fname in [
            "pytorch_model.bin", "model.safetensors"
        ]
    )
    has_adapter = any(
        os.path.exists(os.path.join(model_path, fname)) for fname in [
            "adapter_config.json", "adapter_model.safetensors"
        ]
    )

    if has_full and not has_adapter:
        # Load the fully fine-tuned model directly
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    else:
        # Load base model and then attach adapter (or fall back to base if adapter missing)
        base_model = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            **model_kwargs
        )
        # Align base model embedding size with fine-tuned tokenizer to prevent size mismatch
        try:
            vocab_len = len(tokenizer)
            if base_model.get_input_embeddings().num_embeddings != vocab_len:
                base_model.resize_token_embeddings(vocab_len)
        except Exception as e:
            print(f"Warning: embedding resize failed: {e}")
        if has_adapter:
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            print("Warning: No adapter or full model found in output dir; using base model.")
            model = base_model

    return model, tokenizer

def _extract_thinking_and_clean(reply: str):
    thinking = ""
    lower = reply.lower()
    # HTML-like tags
    if "<think>" in lower and "</think>" in lower:
        try:
            start = lower.find("<think>") + len("<think>")
            end = lower.rfind("</think>")
            thinking = reply[start:end].strip()
            reply = reply[lower.rfind("</think>") + len("</think>"):].strip()
            lower = reply.lower()
        except Exception:
            pass
    # BB-style tags
    if "[thinking]" in lower and "[/thinking]" in lower and not thinking:
        try:
            start = lower.find("[thinking]") + len("[thinking]")
            end = lower.rfind("[/thinking]")
            thinking = reply[start:end].strip()
            reply = reply[lower.rfind("[/thinking]") + len("[/thinking]"):].strip()
            lower = reply.lower()
        except Exception:
            pass
    # Remove leading meta lines
    lines = reply.splitlines()
    filtered = []
    for ln in lines:
        if ln.strip().lower().startswith(("thought:", "reasoning:", "analysis:")):
            thinking += ("\n" if thinking else "") + ln
        else:
            filtered.append(ln)
    reply = "\n".join(filtered).strip()
    return thinking.strip(), reply

def chat(message, history):
    # history is a list of dicts when type="messages" is used
    system_prompt = _CONFIG['personality'].get('system_prompt', '').strip()
    messages = []
    # Enforce explicit thinking delimiters so UI can separate thoughts from the final answer
    messages.append({
        "role": "system",
        "content": (
            "If you need to reason, write your internal thoughts inside <think> and </think> first. "
            "Then, after </think>, provide the final answer only. Do not include internal thoughts in the final answer."
        )
    })
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # Extend with prior history (already role/content dicts)
    for item in history:
        role = item.get('role')
        content = item.get('content')
        if role and content is not None:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    # Emit a pending thinking message first
    pending = ChatMessage(
        content="",
        metadata={"title": "_Thinking_", "id": 0, "status": "pending"}
    )
    yield pending

    try:
        prompt_text = _TOKENIZER.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        prompt_text = (f"{system_prompt}\n\n" if system_prompt else "") + "\n".join(
            f"{m['role'].title()}: {m['content']}" for m in messages
        ) + "\nAssistant:"

    device = getattr(_MODEL, 'device', next(_MODEL.parameters()).device)
    # Stage 1: force thinking by appending <think> and stopping at </think>
    think_start_prompt = prompt_text + "<think>"
    inputs_think = _TOKENIZER(think_start_prompt, return_tensors="pt")
    inputs_think = {k: v.to(device) for k, v in inputs_think.items()}

    class _StopOnString(StoppingCriteria):
        def __init__(self, stop_string: str, tokenizer, input_len: int):
            self.stop_string = stop_string
            self.tokenizer = tokenizer
            self.input_len = input_len
        def __call__(self, input_ids, scores) -> bool:
            text = self.tokenizer.decode(input_ids[0][self.input_len:], skip_special_tokens=False)
            return self.stop_string in text

    stop_criteria = StoppingCriteriaList([
        _StopOnString("</think>", _TOKENIZER, inputs_think['input_ids'].shape[1])
    ])

    streamer_think = TextIteratorStreamer(_TOKENIZER, skip_prompt=True, skip_special_tokens=False)
    gen_kwargs_think = dict(
        **inputs_think,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=_TOKENIZER.eos_token_id,
        pad_token_id=_TOKENIZER.pad_token_id,
        streamer=streamer_think,
        stopping_criteria=stop_criteria,
    )
    thread_think = Thread(target=_MODEL.generate, kwargs=gen_kwargs_think)
    thread_think.start()

    think_buf = ""
    for chunk in streamer_think:
        think_buf += chunk
        # live update thinking panel
        pending.content = think_buf.replace("</think>", "").strip()
        yield pending
    pending.metadata["status"] = "done"
    pending.content = think_buf.replace("</think>", "").strip()
    yield pending

    # Stage 2: generate final answer conditioned on captured thinking
    injected = f"<think>{pending.content}</think>"
    final_prompt = prompt_text + injected
    inputs_final = _TOKENIZER(final_prompt, return_tensors="pt")
    inputs_final = {k: v.to(device) for k, v in inputs_final.items()}

    streamer_final = TextIteratorStreamer(_TOKENIZER, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs_final = dict(
        **inputs_final,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=_TOKENIZER.eos_token_id,
        pad_token_id=_TOKENIZER.pad_token_id,
        streamer=streamer_final,
    )
    thread_final = Thread(target=_MODEL.generate, kwargs=gen_kwargs_final)
    thread_final.start()

    final_msg = ChatMessage(content="")
    final_buf = ""
    for chunk in streamer_final:
        final_buf += chunk
        final_msg.content = final_buf.strip()
        yield [pending, final_msg]
    yield final_msg

def user_message(message, history):
    """Handle user message input"""
    return "", history + [{"role": "user", "content": message}]

def bot_response(history):
    """Generate bot response with image switching"""
    if not history:
        return history, "media/mikasa.png"

    # Get the last user message
    user_msg = history[-1]["content"]

    # Track current image state
    current_image = "media/thinking.gif"

    # Start with thinking image - only set once
    yield history, current_image

    # Process the chat
    chat_generator = chat(user_msg, history[:-1])  # Exclude current user message from history

    thinking_phase = True
    previous_image = current_image

    for response in chat_generator:
        current_history = history[:-1]  # Start with history minus the user message

        if isinstance(response, list):
            # Multiple messages (thinking + final)
            current_history.extend(response)

            # Check if we have both thinking and final messages and need to transition
            if len(response) >= 2:
                final_msg = response[1]
                # Transition from thinking to talking when final message has substantial content
                if (hasattr(final_msg, 'content') and final_msg.content.strip() and
                    thinking_phase):
                    thinking_phase = False
                    current_image = "media/talking.gif"

        else:
            # Single message - convert to proper format
            if hasattr(response, 'content'):
                current_history.append({"role": "assistant", "content": response.content})
            else:
                current_history.append({"role": "assistant", "content": str(response)})

        # Only yield new image if it changed, otherwise use previous image
        if current_image != previous_image:
            yield current_history, current_image
            previous_image = current_image
        else:
            # Keep the same image to prevent GIF restart
            yield current_history, previous_image

    # Final state - switch to static image when completely done
    final_history = current_history if 'current_history' in locals() else history
    yield final_history, "media/mikasa.png"

def launch_ui():
    title = "Mikasa ✨"

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    type="messages",
                    height=500
                )
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    container=False
                )
                clear = gr.Button("Clear")

                # Examples
                gr.Examples(
                    examples=[
                        "What's your name?",
                        "Write a cute haiku about friendship",
                        "Translate to Japanese: I will always support you",
                    ],
                    inputs=msg
                )

            with gr.Column(scale=1):
                mikasa_image = gr.Image(
                    value="media/mikasa.png",
                    label="Mikasa",
                    height=400,
                    interactive=False,
                    show_download_button=False
                )

        # Event handlers
        msg.submit(
            user_message,
            [msg, chatbot],
            [msg, chatbot],
            queue=False
        ).then(
            bot_response,
            chatbot,
            [chatbot, mikasa_image]
        )

        clear.click(
            lambda: ([], "media/mikasa.png"),
            None,
            [chatbot, mikasa_image],
            queue=False
        )

    demo.launch()

def main():
    global _CONFIG, _MODEL, _TOKENIZER
    _CONFIG = load_config()
    print(f"\nPlatform: {platform.system()}")
    print(f"PyTorch version: {torch.__version__}")
    if platform.system() == "Darwin":
        print(f"MPS available: {torch.backends.mps.is_available()}")
    _MODEL, _TOKENIZER = load_model_and_tokenizer(_CONFIG)
    print("\n✨ Mikasa is ready! Launching Gradio... ✨\n")
    launch_ui()

if __name__ == "__main__":
    main()