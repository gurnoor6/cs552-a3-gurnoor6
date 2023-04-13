
import torch
from typing import Any, Dict
from a3_utils import *
import json

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)

class GreedySearchDecoderForT5(GeneratorForT5):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer):
        super().__init__(model, tokenizer)
    
    def search(
        self,
        inputs: dict,
        max_new_tokens: int
    ) -> torch.LongTensor:
        """Generates sequences of token ids for T5ForConditionalGeneration 
        (which has a language modeling head) using greedy decoding. 
        This means that we always pick the next token with the highest score/probability.

        This function always does early stopping and does not handle the case 
        where we don't do early stopping. 
        It also only handles inputs of batch size = 1.

        Inherits variables and helper functions from GeneratorForT5().

        Args:
            inputs (dict): the tokenized input dictionary returned by the T5 tokenizer
            max_new_tokens (int): a limit for the amount of decoder outputs 
                                  we desire to generate

        Returns:
            torch.LongTensor: greedy decoded best sequence made of token ids of size (1,generated_seq_len)
                              This should include the starting pad token!
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(inputs, max_new_tokens)
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
        ########################################################################

        ########################################################################
        # TODO: Implement me! Read the docstring above and this comment carefully.
        #
        # For greedy decoding, keep in mind of the following:
        #   - do not handle input batch size != 1.
        #   - return the sampled sequence as it is (not in a dictionary).
        #     You should not return a score you get for the sequence.
        #   - always do early stopping: this means that if the next token is an EOS
        #     (end-of-sentence) token, you should stop decoding.
        #   - you might want to use the self.prepare_next_inputs function inherited
        #     by this class as shown here:
        #
        #       First token use: 
        #           model_inputs = self.prepare_next_inputs(model_inputs=inputs)
        #       Future use: 
        #           model_inputs = self.prepare_next_inputs(
        #               model_inputs = model_inputs,
        #               new_token_id = new_token_id,
        #           )
        ########################################################################

        new_tokens = []
        i = 0
        while True:
            if i == 0:
                model_inputs = self.prepare_next_inputs(model_inputs = inputs)
            else:
                model_inputs = self.prepare_next_inputs(model_inputs = model_inputs, new_token_id = new_tokens[-1])
            
            x = self.model(**model_inputs)
            new_tokens.append(torch.argmax(x.logits, dim=2)[-1][-1])
            i += 1
            
            if new_tokens[-1].item() == self.eos_token_id:
                model_inputs["decoder_input_ids"] = torch.cat([
                    model_inputs["decoder_input_ids"],
                    model_inputs["decoder_input_ids"].new_ones((model_inputs["decoder_input_ids"].shape[0], 1))],
                    dim=-1,
                )
                model_inputs["decoder_input_ids"][0][-1] = self.eos_token_id
                break
            
            if i == max_new_tokens + 1:
                break
        
        return model_inputs["decoder_input_ids"]
        
        


class BeamSearchDecoderForT5(GeneratorForT5):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer):
        super().__init__(model, tokenizer)
    
    def beam_search(
        self,
        model_inputs,
        p_values,
        sequences,
        max_new_tokens,
        num_beams,
        length_penalty=0.0
    ):
        print("new:", max_new_tokens)
        # print(max_new_tokens)
        if max_new_tokens == 0:
            return p_values, sequences

        new_sequences = []
        new_p = []
        for p_initial, sequence in zip(p_values, sequences):
            if sequence[-1].item() == self.eos_token_id:
                new_sequences.append(sequence)
                new_p.append(p_initial)
                continue

            model_inputs_copy = model_inputs.copy()
            
            is_pad_token = True
            for item in sequence:
                if is_pad_token:
                    is_pad_token = False
                    continue
                model_inputs = self.prepare_next_inputs(model_inputs = model_inputs, new_token_id = item)


            x = self.model(**model_inputs)
            logits = x.logits[-1, -1, :]
            p = torch.log(torch.exp(logits) / torch.sum(torch.exp(logits)))
            p_sort, indices = torch.sort(p, descending=True)
            p_sort = p_sort[:num_beams]
            max_items = indices[:num_beams]
            for p_value, item in zip(p_sort, max_items):
                new_sequences.append(torch.cat((sequence, torch.unsqueeze(item, dim=0))))

                # penalty factor
                pen_f = (len(sequences[0]) ** length_penalty)
                new_p.append(p_initial * pen_f + p_value)

            model_inputs = model_inputs_copy

        new_p, new_sequences = torch.tensor(new_p), torch.stack(new_sequences)

        # # adding length penalty
        new_p /= (len(new_sequences[0]) ** length_penalty)

        p_sort, indices = torch.sort(new_p, descending=True)
        p_sort = p_sort[:num_beams]
        new_sequences = new_sequences[indices][:num_beams]

        return self.beam_search(model_inputs, p_sort, new_sequences, max_new_tokens-1, num_beams, length_penalty)
        
    
    def search(
        self,
        inputs,
        max_new_tokens: int,
        num_beams: int,
        num_return_sequences=1,
        length_penalty: float = 0.0
    ) -> dict: 
        """Generates sequences of token ids for T5ForConditionalGeneration 
        (which has a language modeling head) using beam search. 
        This means that we sample the next token according to the best conditional 
        probabilities of the next beam_size tokens.

        This function always does early stopping and does not handle the case 
        where we don't do early stopping. 
        It also only handles inputs of batch size = 1 and of beam size > 1 
            (1=greedy search, but you don't have to handle it)
        
        It also include a length_penalty variable that controls the score assigned to a long generation.
        Implemented by exponiating the length of the decoder inputs to this value. 
        This is then used to divide the score which can be calculated as the sum of the log probabilities so far.

        Inherits variables and helper functions from GeneratorForT5().

        Args:
            inputs (_type_): the tokenized input dictionary returned by the T5 tokenizer
            max_new_tokens (int): a limit for the amount of decoder outputs 
                                  we desire to generate
            num_beams (int): number of beams for beam search
            num_return_sequences (int, optional):
                the amount of best sequences to return. Cannot be more than beam size.
                Defaults to 1.
            length_penalty (float, optional): 
                exponential penalty to the length that is used with beam-based generation. 
                It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. 
                Defaults to 0.0.

        Returns:
            dict: dictionary with two key values:
                    - "sequences": torch.LongTensor depicting the best generated sequences (token ID tensor) 
                        * shape (num_return_sequences, maximum_generated_sequence_length)
                        * ordered from best scoring sequence to worst
                        * if a sequence has reached end of the sentence, 
                          you can fill the rest of the tensor row with the pad token ID
                    - "scores": length penalized log probability score list, ordered by best score to worst
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(
            inputs, 
            max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences
        )
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
        ########################################################################

        ########################################################################
        # TODO: Implement me! Read the docstring above and this comment carefully.
        #
        # Given a probability distribution over the possible next tokens and 
        # a beam width (here num_beams), needs to keep track of the most probable 
        # num_beams candidates.
        # You can do so by keeping track of the sum of the log probabilities of 
        # the best num_beams candidates at each step.
        # Then recursively repeat this process until either:
        #   - you reach the end of the sequence
        #   - or you reach max_length
        #
        # For beam search, keep in mind of the following:
        #   - do not handle input batch size != 1.
        #   - always do early stopping: this means that if the next token is an EOS
        #     (end-of-sentence) token, you should stop decoding.
        #   - don't forget to implement the length penalty
        #   - you might want to use the self.prepare_next_inputs function inherited
        #     by this class as shown here:
        #
        #       First token use: 
        #           model_inputs = self.prepare_next_inputs(model_inputs=inputs)
        #       Future use: 
        #           model_inputs = self.prepare_next_inputs(
        #               model_inputs = model_inputs,
        #               new_token_id = new_token_id,
        #           )
        ########################################################################
        model_inputs = self.prepare_next_inputs(model_inputs = inputs)
        x = self.model(**model_inputs)
        logits = x.logits[-1, -1, :]
        p = torch.log(torch.exp(logits) / torch.sum(torch.exp(logits)))
        p_sort, indices = torch.sort(p, descending=True)
        max_items = indices[:num_beams]
        max_items = list(map(lambda x: torch.tensor(
                                            [self.model.config.pad_token_id,
                                            x.item()]
                                        ), max_items))
        max_items = torch.stack(max_items)
        p_sort = p_sort[:num_beams] / (len(max_items[0]) ** length_penalty)
        
        p_values, sequences = self.beam_search(model_inputs, p_sort, max_items, max_new_tokens-1, num_beams, length_penalty)
        return {
            "sequences": sequences[:num_return_sequences],
            "scores": p_values[:num_return_sequences]
        }


def main():
    ############################################################################
    # NOTE: You can use this space for testing but you are not required to do so!
    ############################################################################
    seed = 421
    torch.manual_seed(seed)
    torch.set_printoptions(precision=16)
    model_name = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # 2) Load relevant inputs
    with open("part1_input_data.json", 'r') as read_file:
        input_data = json.load(read_file)

    t5_paper_abstract = input_data["t5_paper_abstract"]
    summary_prefix = "summarize: "
    abstract_inputs = tokenizer(
        [summary_prefix + t5_paper_abstract], 
        max_length=MAX_T5_SEQ_LENGTH, 
        truncation=True, 
        return_tensors="pt"
    )
    beam_decoder = BeamSearchDecoderForT5(model=model, tokenizer=tokenizer)
    result_dict = beam_decoder.search(
        inputs=abstract_inputs,
        max_new_tokens=2,
        num_beams=4,
        length_penalty=0.0,
        num_return_sequences=4,
    )

    print(result_dict['scores'])

if __name__ == '__main__':
    main()