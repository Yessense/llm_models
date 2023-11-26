from PIL import Image
import torch

from transformers import StoppingCriteriaList
import sys
MOUNTED_FOULDER = "/root/.cache/huggingface/hub"
minigpt4_path = f'{MOUNTED_FOULDER}/MiniGPT-4'
if sys.path[-1] != minigpt4_path:
    sys.path.append(minigpt4_path)
from minigpt4.conversation.conversation import StoppingCriteriaSub, Conversation, SeparatorStyle

class MiniGPT4Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        stop_words_ids = [torch.tensor([835]).to(self.device),
                        torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        self.conv, self.img_list = None, None
        self.reset_history()
        
    def ask(self, text):
        if len(self.conv.messages) > 0 and self.conv.messages[-1][0] == self.conv.roles[0] \
                and self.conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            self.conv.messages[-1][1] = ' '.join([self.conv.messages[-1][1], text])
        else:
            self.conv.append_message(self.conv.roles[0], text)

    def answer(self, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
            repetition_penalty=1.0, length_penalty=1, temperature=0.0, max_length=3500):
        self.conv.append_message(self.conv.roles[1], None)
        embs = self.get_context_emb(self.img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True if num_beams==1 else False,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        self.conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def upload_img(self, image):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        image_emb, _ = self.model.encode_img(image)
        self.img_list.append(image_emb)
        self.conv.append_message(self.conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        return msg

    def upload_images(self, images):
        for image in images:
            if isinstance(image, str):  # is a image path
                raw_image = Image.open(image).convert('RGB')
                image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
            elif isinstance(image, Image.Image):
                raw_image = image
                image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
            elif isinstance(image, torch.Tensor):
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                image = image.to(self.device)
            image_emb, _ = self.model.encode_img(image)
            self.img_list.append(image_emb)
        
        msg = "Received."
        return msg

    def get_context_emb(self, img_list):
        prompt = self.conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
    
    def reset_history(self):
        self.conv = Conversation(
            system="Give the following image: <Img>ImageContent</Img>. "
                "You will be able to see the image once I provide it to you. Please answer my questions.",
            roles=("Human", "Assistant"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="###",
        )
        self.img_list = []
