# import json
import torch
import os
import numpy as np
#import fairseq
import skimage.measure
import argparse
import soundfile as sf
from shutil import copyfile
from npy_append_array import NpyAppendArray
import tqdm
#from omegaconf import OmegaConf
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
# from fairseq.models.hubert.hubert import HubertConfig, HubertModel
import torchaudio

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data", help="Path with .tsv files pointing to the audio data") # ~/../data_mundus/clarity_CPC2_data_16k/clarity_data/HA_outputs/train.1/
    parser.add_argument("--split", help="Which split", required=True) 
    parser.add_argument("--save-dir", help="Output path to store the features", required=True)
    # parser.add_argument("--checkpoint", help="Path to the WavLM checkpoint", required=True)
    return parser

def encode(wave, sr, model, processor, device='cuda'):
  inputs = processor(wave, sampling_rate=sr, return_tensors="pt", padding=True).to(device)
  with torch.no_grad():
    outputs = model(**inputs)
  return outputs

def get_iterator(args, mdl, processor, encode, device):
    with open(os.path.join(args.data, args.split) + ".tsv", "r") as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        files = [os.path.join(root, line.split("\t")[0]) for line in lines if len(line) > 0]
        num = len(files)

        def iterate():
            for fname in files:
                # feats = reader.get_feats(fname)
                # wav, sr = sf.read(fname)
                wav, sr = torchaudio.load(fname)
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)  # wav2vev2 bert2.0 feature extractor was trained on 16000 sampling rate
                # wav = torch.from_numpy(wav).float().cuda()
                if wav.dim() > 1:
                    feats = []
                    for d in range(wav.dim()):
                        # m_in = wav[:, d].view(-1, 1)
                        m_in = wav[d, :].view(1, -1)
                        with torch.no_grad():
                            # if cfg.task.normalize:
                            #     m_in = torch.nn.functional.layer_norm(m_in , m_in.shape)
                            # audio_rep = mdl(source=m_in, mask=False, features_only=True, output_layer=cfg.model.encoder_layers)["layer_results"]
                            # audio_rep = [rep[0] for rep in audio_rep]  # most probably taking embeds from all heads
                            # audio_rep = torch.concatenate(audio_rep, dim=1).transpose(1, 0) # we did the same thing already with embeds.hidden_state + torch.cat
                            audio_rep = encode(wav, 16000, mdl, processor, device)   # get the embeddings
                            audio_rep = torch.cat(audio_rep.hidden_states, dim=0) # get hidden states of all heads
                            audio_rep = audio_rep.cpu().numpy() # ok
                        audio_rep = skimage.measure.block_reduce(audio_rep, (1, 20, 1), np.mean) # downsample x20 (temporal pooling)

                        audio_rep = np.transpose(audio_rep, (1, 0, 2)) # [time, heads, width]
                        feats.append(audio_rep)
                    yield np.concatenate([np.expand_dims(feats[0], axis=1),
                                          np.expand_dims(feats[1], axis=1)], axis=1)
                else:
                    wav = wav.view(1, -1)
                    with torch.no_grad():
                        # if cfg.task.normalize:
                        #     wav = torch.nn.functional.layer_norm(wav , wav.shape)
                        audio_rep = encode(wav, 16000, mdl, processor, device)
                        audio_rep = torch.cat(audio_rep.hidden_states, dim=0)
                        audio_rep = audio_rep.cpu().numpy()
                    audio_rep = skimage.measure.block_reduce(audio_rep, (1, 20, 1), np.mean) # downsample x20
                    audio_rep = np.transpose(audio_rep, (1, 0, 2))
                    yield audio_rep
    return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    def create_files(dest):
        copyfile(os.path.join(args.data, args.split) + ".tsv", dest + ".tsv")
        if os.path.exists(os.path.join(args.data, args.split) + ".itl"):
            copyfile(os.path.join(args.data, args.split) + ".itl", dest + ".itl")
        if os.path.exists(os.path.join(args.data, args.split) + ".lis"):
            copyfile(os.path.join(args.data, args.split) + ".lis", dest + ".lis")

        if os.path.exists(dest + ".npy"):
            os.remove(dest + ".npy")
        npaa = NpyAppendArray(dest + ".npy")
        return npaa

    save_path = os.path.join(args.save_dir, args.split)
    npaa = create_files(save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda"
    print(device)
    # checkpoint_path = args.checkpoint
    # model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    #     [checkpoint_path]
    # )
    # model = model[0]

    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # cfg = OmegaConf.create(checkpoint["cfg"])
    #if cfg.model._name == "hubert_ctc":
    #    cfg = cfg.model.w2v_args
    #    # Create a new state_dict with modified keys
    #    state_dict = {}
    #    prefix = "w2v_encoder.w2v_model."

    #    for key, value in checkpoint['model'].items():
    #        if key.startswith(prefix):
    #            new_key = key[len(prefix):]
    #            state_dict[new_key] = value
    #    cfg.model["layer_type"] = "transformer"
    #    cfg.model["required_seq_len_multiple"] = 1
    #else:
    #    state_dict = checkpoint['model']

    # model = HubertModel(HubertConfig.from_namespace(cfg.model), cfg.task, [None])
    # model.load_state_dict(state_dict, strict=False)

    processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", output_attentions=False, output_hidden_states=True)

    model.eval()
    model.to(device)

    generator, num = get_iterator(args, model, processor, encode, device)
    iterator = generator()

    with open(save_path + ".lengths", "w") as l_f:
        for feats in tqdm.tqdm(iterator, total=num):
            if len(feats.shape) == 2:
                feats = np.repeat(np.expand_dims(feats, axis=1), repeats=2, axis=1)
            print(len(feats), file=l_f)

            if len(feats) > 0:
                npaa.append(np.ascontiguousarray(feats))
    del model


if __name__ == "__main__":
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 7310))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()
    main()
