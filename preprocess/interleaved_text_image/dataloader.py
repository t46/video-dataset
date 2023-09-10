# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Build a StreamingTextDataset dataset and dataloader for training."""

import os
from itertools import islice
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import transformers
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
import argparse
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import itertools


def build_tokenizer(om_tokenizer_config: DictConfig) -> PreTrainedTokenizerBase:
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    resolved_om_tokenizer_config = om.to_container(om_tokenizer_config,
                                                   resolve=True)
    tokenizer_kwargs = resolved_om_tokenizer_config.get(  # type: ignore
        'kwargs', {})
    tokenizer_name = resolved_om_tokenizer_config['name']  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                              **tokenizer_kwargs)

    # HuggingFace does not respect the model_max_length kwarg, and overrides it with
    # min(kwargs['model_max_length'], original_config['model_max_length']), so we
    # explicitly set it here
    tokenizer.model_max_length = tokenizer_kwargs.get(
        'model_max_length',
        int(1e30),
    )

    return tokenizer

class StreamingInterleavedDataset(StreamingDataset):
    """Generic text dataset using MosaicML's StreamingDataset.

    Args:
        tokenizer (Tokenizer): HuggingFace tokenizer to
            tokenize samples.
        max_seq_len (int): The max sequence length of each sample.
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from,
            which may be upsampled or downsampled. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            `False``.
        epoch_size (int, optional): Number of samples to draw per epoch balanced across all
            streams. If ``None``, takes its value from the total number of underlying samples.
            Provide this field if you are weighting streams relatively to target a larger or
            smaller epoch size. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        cache_limit (Union[int, str], optional) - Maximum size in bytes of this StreamingDataset's
            shard cache. Before downloading a shard, the least recently used resident shard(s) may
            be evicted (deleted from the local cache) in order to stay under the limit. Set to None
            to disable shard eviction. Supports integer bytes as well as string human-readable
            bytes (e.g., 100b, 64kb, 77mb, and so on). Defaults to None.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. Defaults to ``None``, which is interpreted as the number of nodes of the
            initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1b``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int): Unit of shuffle. Defaults to ``1 << 18``.
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 max_seq_len: int,
                 streams: Optional[Sequence[Stream]] = None,
                 remote: Optional[str] = None,
                 local: Optional[str] = None,
                 split: Optional[str] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 keep_zip: bool = False,
                 epoch_size: Optional[int] = None,
                 predownload: int = 100_000,
                 cache_limit: Optional[Union[int, str]] = None,
                 partition_algo: str = 'orig',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1b',
                 shuffle_seed: int = 9176,
                 shuffle_block_size: int = 1 << 18,
                 chunk_size: int = 5,
                 **kwargs: Any):

        group_method = kwargs.pop('group_method', None)
        if group_method is not None:
            raise NotImplementedError(
                'group_method is deprecated and has been removed.\nTo ' +
                'concatenate, use the --concat_tokens ' +
                'argument when creating your MDS dataset with concat_c4.py')

        if len(kwargs) > 0:
            raise ValueError(
                f'StreamingTextDataset() got an unexpected keyword argument: {kwargs}'
            )

        # NOTE: We dno't specify the split.
        # if local is not None and (remote is None or (local == remote)):
        #     if os.path.isdir(local):
        #         contents = set(os.listdir(local))
        #         if split not in contents:
        #             raise ValueError(
        #                 f'local directory {local} does not contain split {split}'
        #             )

        # Build Dataset
        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=validate_hash,
            keep_zip=keep_zip,
            epoch_size=epoch_size,
            predownload=predownload,
            cache_limit=cache_limit,
            partition_algo=partition_algo,
            num_canonical_nodes=num_canonical_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_seed=shuffle_seed,
            shuffle_block_size=shuffle_block_size,
        )
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.chunk_size = chunk_size

    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample: Mapping):
        if self.tokenizer._pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            raise RuntimeError(
                'If tokenizing on-the-fly, tokenizer must have a pad_token_id')

        return self.tokenizer(text_sample['text'],
                              truncation=True,
                              padding='max_length',
                              max_length=self.max_seq_len)

    def _read_binary_tokenized_sample(self, sample: Dict[str, Any]):
        return torch.from_numpy(
            np.frombuffer(sample['tokens'],
                          dtype=np.int64)[:self.max_seq_len].copy())
    
    def _read_binary_data(self, sample):
        return torch.from_numpy(
            np.frombuffer(sample,
                          dtype=np.int64).copy())

    # How to process a sample
    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        # NOTE: I use only tokenized samples.
        # if 'text' in sample:
        #     token_sample = self._tokenize(sample)
        # elif 'tokens' in sample:
        #     token_sample = self._read_binary_tokenized_sample(sample)
        # else:
        #     raise RuntimeError(
        #         'StreamingTextDataset needs samples to have a `text` or `tokens` column'
        #     )
        if ('frame' not in sample) or ('subtitle_data' not in sample):
            raise RuntimeError(
                'StreamingTextDataset needs samples to have both `frame` and `subtitle_data` column'
            )
        frames = self._read_binary_data(sample.get('frame', None))  # torch.tensor

        # Chunk frames (I don't chunk the subtitles.)
        num_chunks, remainder = divmod(len(frames), self.chunk_size)
        if remainder != 0:
            chunked_frames = torch.stack(torch.chunk(frames[:-remainder], num_chunks, dim=0), dim=0)  # (num_frames, H, W, C) -> (num_chunks, chunk_size, H, W, C)
        else:
            chunked_frames = torch.stack(torch.chunk(frames, num_chunks, dim=0), dim=0)

        subtitle_data = sample.get('subtitle_data', None)
        subtitles = torch.tensor(list(itertools.chain.from_iterable(subtitle_data['subtitle'])))  # torch.tensor

        return (subtitles, chunked_frames)


# Multimodal Collate Function
class MultimodalCollateWrapper:
    def __init__(self, text_collator, image_collator, video_collator, audio_collator, multimodal_position_ids_collator) -> None:
        self.text_collator = text_collator
        self.image_collator = image_collator
        # self.video_collator = video_collator
        # self.audio_collator = audio_collator
        self.multimodal_position_ids_collator = multimodal_position_ids_collator
    
    def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
        # Convert examples (list of tuples) to a list of lists (one list per modality)
        examples = list(zip(*examples))
        batch = self.text_collator(examples[0])  # NOTE: subtitles
        batch['images'] = self.image_collator(examples[1])  # NOTE: frames
        # batch['video'] = self.video_collator(examples[2])
        # batch['audio'] = self.audio_collator(examples[3])
        # batch['multimodal_position_ids'] = self.multimodal_position_ids_collator(examples[2])  # TODO: implement this
        return batch


# NOTE: We don't use this for now.
# class ConcatenatedSequenceCollatorWrapper:
#     """Collator wrapper to add sequence_id to batch."""

#     def __init__(
#         self,
#         base_collator: Callable,
#         eos_token_id: Optional[int] = None,
#         bos_token_id: Optional[int] = None,
#     ):
#         self.base_collator = base_collator
#         if (eos_token_id is None) and (bos_token_id is None):
#             raise ValueError(
#                 'Must supply a value for either eos_token_id or bos_token_id, but got None for both.'
#             )
#         if (eos_token_id is not None) and (bos_token_id is not None):
#             raise ValueError(
#                 'Cannot use *both* EOS and BOS tokens for detecting sequence boundaries. ' +\
#                 'Please supply `eos_token_id` if sequences end with an EOS token, or use ' +\
#                 '`bos_token_id` if sequences start with a BOS token.'
#             )

#         self.split_token_id = eos_token_id
#         self.bos_mode = False
#         if eos_token_id is None:
#             self.split_token_id = bos_token_id
#             self.bos_mode = True

#     def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
#         batch = self.base_collator(examples)
#         batch['sequence_id'] = self.get_sequence_id_from_batch(batch)
#         return batch

#     def get_sequence_id_from_batch(
#             self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        
#         multimodal_position_ids = batch['multimodal_position_ids'] # TODO
#         number_of_modalities = multimodal_position_ids.shape[0]
#         batch_size = multimodal_position_ids.shape[1]
#         max_seq_len = multimodal_position_ids.shape[2]
#         tokens = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=multimodal_position_ids.device)
#         token_ids = batch['input_ids']
        
#         # tokens[multimodal_position_ids[:, 0, :]] = token_ids
#         tokens[torch.arange(tokens.size(0))[:,None], multimodal_position_ids[:, 0, :]] = token_ids
#         is_separator = torch.eq(tokens,
#                                 self.split_token_id)  # type: ignore
#         cumulative_sep = torch.cumsum(is_separator,
#                                       dim=1).to(batch['input_ids'].dtype)
#         # If separator token is bos, we're already done
#         if self.bos_mode:
#             return cumulative_sep

#         # If separator token is eos, right shift 1 space
#         left_zeros = cumulative_sep.new_zeros((cumulative_sep.shape[0], 1))
#         return torch.cat([left_zeros, cumulative_sep[:, :-1]], dim=1)


class TextNeoXCollateWrapper:

    def __init__(self, collator) -> None:
        self.collator = collator
    
    def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
        batch = self.collator(examples)
        batch['text'] = batch['input_ids']
        return batch

class PaddedCollateWrapper:
    def __init__(self, pad_token_id, take_transpose=False) -> None:
        self.pad_token_id = pad_token_id
        self.take_transpose = take_transpose

    def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
        if self.take_transpose:
            # Apply transpose to each example in batch parallely using map function
            examples = list(map(lambda x: x.transpose(0, 1), examples))

        batch = torch.nn.utils.rnn.pad_sequence(examples, batch_first=True, padding_value=self.pad_token_id)

        if self.take_transpose:
            batch = batch.transpose(1, 2)
        return batch
    
def build_interleaved_dataloader(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: int,
):
    assert cfg.name == 'text', f'Tried to build text dataloader with cfg.name={cfg.name}'
    if cfg.dataset.get('group_method', None) is not None:
        raise NotImplementedError(
            'group_method is deprecated and has been removed.\nTo ' +
            'concatenate, use the --concat_tokens ' +
            'argument when creating your MDS dataset with convert_dataset_hf.py'
        )

    # get kwargs
    streams_dict = cfg.dataset.pop('streams', None)
    mlm_probability = cfg.dataset.pop('mlm_probability', None)
    eos_token_id = cfg.dataset.pop('eos_token_id', None)
    bos_token_id = cfg.dataset.pop('bos_token_id', None)

    # build streams
    streams = None
    if streams_dict is not None:
        streams = []
        for _, stream in streams_dict.items():
            # stream is the streams kwargs
            # fwd all kwargs with **stream allows streaming to check args
            streams.append(Stream(**stream))

    # build dataset potentially with streams
    dataset = StreamingInterleavedDataset(
        tokenizer=tokenizer,
        streams=streams,
        **cfg.dataset,
    )

    # NOTE: We already tokenized the text in the dataset, so we don't need to do it again.
    text_collate_fn = transformers.DataCollatorForLanguageModeling(
        tokenizer=dataset.tokenizer,
        mlm=mlm_probability is not None,
        mlm_probability=mlm_probability)

    text_collate_fn = TextNeoXCollateWrapper(text_collate_fn)
    
    image_collate_fn = PaddedCollateWrapper(pad_token_id=-1) # Each sample: (num_chunks, chunk_size, H, W, C)

    multimodal_position_ids_collate_fn = PaddedCollateWrapper(pad_token_id=-1, take_transpose=True) # Each sample: (num_modalities, max_seq_len)
    
    collate_fn = MultimodalCollateWrapper(text_collator=text_collate_fn, 
                                          image_collator=image_collate_fn, 
                                          video_collator=None, 
                                          audio_collator=None, 
                                          multimodal_position_ids_collator=multimodal_position_ids_collate_fn)

    # NOTE: We don't use this.  
    # if (eos_token_id is not None) or (bos_token_id is not None):
    #     # Note: Will raise an error if both are non-None
    #     collate_fn = ConcatenatedSequenceCollatorWrapper(
    #         base_collator=collate_fn,
    #         eos_token_id=eos_token_id,
    #         bos_token_id=bos_token_id)

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', True),
        prefetch_factor=cfg.get('prefetch_factor', 2),
        persistent_workers=cfg.get('persistent_workers', True),
        timeout=cfg.get('timeout', 0),
    )


# Helpful to test if your dataloader is working locally
# Run `python data.py  --local_path [local] [--remote_path remote, optional]` and verify that batches are printed out
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer',
                        type=str,
                        default='EleutherAI/gpt-neox-20b',
                        help='the name of the tokenizer to use')
    parser.add_argument('--local_path',
                        type=str,
                        required=True,
                        help='the path to the local copy of the dataset')
    parser.add_argument(
        '--remote_path',
        type=str,
        default=None,
        help='the path to the remote copy to stream from (optional)')
    # NOTE: DO NOT USE THIS.
    # parser.add_argument('--split',
    #                     type=str,
    #                     default='val',
    #                     help='which split of the dataset to use')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=32,
                        help='max sequence length to test')

    args = parser.parse_args()

    if args.remote_path is not None:
        print(
            f'Reading data from {args.local_path} <- streamed from <- {args.remote_path}'
        )
    else:
        print(f'Reading data from {args.local_path}')

    cfg = {
        'name': 'text',
        'dataset': {
            'local': args.local_path,
            'remote': args.remote_path,
            'shuffle': False,
            'max_seq_len': args.max_seq_len,
            'keep_zip': True,  # in case we need compressed files after testing
        },
        'drop_last': False,
        'num_workers': 4,
    }
    cfg = om.create(cfg)
    device_batch_size = 2

    tokenizer_cfg = {'name': args.tokenizer, 'kwargs': {}}
    tokenizer_cfg['kwargs'] = {'model_max_length': args.max_seq_len}
    tokenizer_cfg = om.create(tokenizer_cfg)
    tokenizer = build_tokenizer(tokenizer_cfg)

    loader = build_interleaved_dataloader(cfg, tokenizer, device_batch_size)
    tokenizer = loader.dataset.tokenizer  # type: ignore
    for batch_ix, batch in enumerate(islice(loader, 5)):
        print('\n')
        print('#' * 20, f'Batch {batch_ix}', '#' * 20)
        for k, v in batch.items():
            print(k, v.shape, v.dtype)
        for sample_ix, token_sample in enumerate(batch['input_ids']):
            print('-' * 20, f' Sample {sample_ix} ', '-' * 20)
            print(tokenizer.decode(token_sample))