
import torch
from typing import Any, Dict
from a3_utils import *

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

        pass

class BeamSearchDecoderForT5(GeneratorForT5):
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

        pass


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


if __name__ == '__main__':
    main()