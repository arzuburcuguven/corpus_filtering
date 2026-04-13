from abc import ABC, abstractmethod
from typing import final
from conllu import TokenList


# ######## helpers



# def get_clause_head_id(token_id, sent):
#     id_map = {t["id"]: t for t in sent if isinstance(t["id"], int)}
#     current_id = sent[id_map]["head"]
#     while current_id != 0 and sent[current_id]["upos"] not in ("VERB", "AUX", "PART"):
#         current_id = sent[current_id]["head"]
#     return current_id

# def head_lemma(token_id, sent):
#     current_id = sent[token_id]["head"]
#     while current_id != 0 and sent[current_id]["upos"] not in ("VERB", "AUX", "PART"):
#         current_id = sent[current_id]["head"]
#     return sent[current_id]["lemma"]


# def lookup(sent, lemmas = None, forms = None, exclude_lemmas = None):
#     for token in sent:
#         if not isinstance(token["id"], int):  # skip range tokens in logic
#             continue
#         if lemmas and token["lemma"] not in lemmas:
#             continue
#         if forms and token["form"] not in forms:
#             continue
#         if token["lemma"] in exclude_lemmas:
#             continue
#         yield token
        
            

class CorpusFilter(ABC):
    
    @abstractmethod
    def _exclude_sent(self, sent: TokenList) -> bool:
        """Return True if sentence contains the target phenomenon."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the phenomenon this filter targets."""
        ...

class NotFilter(CorpusFilter):
    
    @property
    def name(self) -> str:
        return "negation"
    
    def _exclude_sent(self, sent: TokenList) -> bool:
        return any(token["form"].lower() == "not" for token in sent)
    
class ExistentialThereQuantifierFilter(CorpusFilter):

    quantifiers = [
        "a",
        "an",
        "no",
        "some",
        "few",
        "many",
        "all",
        "most",
        "every",
        "each",
    ]
    
    @property
    def name(self) -> str:
        return "existential-there-quantifier"
    
    def _exclude_sent(self, sent: TokenList) -> bool:
        id_map = {t["id"]: t for t in sent if isinstance(t["id"], int)}
        # set 1: copulas which are the head of an expletive there
        there_copulas = set()
        # set 2: verbs whose subjects are the heads of a quantifier
        quantifier_head_head_verbs = set()
        for tok in id_map.values():
            head_id = tok["head"]
            head = id_map.get(head_id)
            
            if tok["lemma"].lower() == "there" and tok["deprel"] == "expl" \
            and head is not None and head["lemma"] == "be":
                there_copulas.add(head["id"])
                #if there_copulas:
                    #print("found there:", sent.metadata.get("text"), (head["id"]))
            # look for members of second set
            if tok["lemma"] in self.quantifiers and head is not None \
            and head["deprel"] is not None and head["deprel"].startswith("nsubj"):
                quantifier_head_head_verbs.add(head["head"])
            

                #if quantifier_head_head_verbs:
                    #print("found Q:", sent.metadata.get("text"), (head["head"]))
                            
        return bool(there_copulas & quantifier_head_head_verbs)

                
    
class ExistentialThereQuantifierFilter(CorpusFilter):

    quantifiers = [
        "a",
        "an",
        "no",
        "some",
        "few",
        "many",
        "all",
        "most",
        "every",
        "each",
    ]
    
    @property
    def name(self) -> str:
        return "existential-there-quantifier"
    
    def _exclude_sent(self, sent: TokenList) -> bool:
        id_map = {t["id"]: t for t in sent if isinstance(t["id"], int)}
        # set 1: copulas which are the head of an expletive there
        there_copulas = set()
        # set 2: verbs whose subjects are the heads of a quantifier
        quantifier_head_head_verbs = set()
        for tok in id_map.values():
            head_id = tok["head"]
            head = id_map.get(head_id)
            
            if tok["lemma"].lower() == "there" and tok["deprel"] == "expl" \
            and head is not None and head["lemma"] == "be":
                there_copulas.add(head["id"])
                #if there_copulas:
                    #print("found there:", sent.metadata.get("text"), (head["id"]))
            # look for members of second set
            if tok["lemma"] in self.quantifiers and head is not None \
            and head["deprel"] is not None and head["deprel"].startswith("nsubj"):
                quantifier_head_head_verbs.add(head["head"])
            

                #if quantifier_head_head_verbs:
                    #print("found Q:", sent.metadata.get("text"), (head["head"]))
                            
        return bool(there_copulas & quantifier_head_head_verbs)

                
class BindingReflexive(CorpusFilter):

    @property
    def name(self) -> str:
        return "Binding-reflexive"

    def _exclude_sent(self, sent: TokenList) -> bool:
        """
        Exclude a sentence if it contains a reflexive pronoun which has a co-indexed nsubj with a relative clause.

        Args:
            sent: A stanza `Sentence` object that has been annotated with dependency
            relations.

        Returns:
            True if the sentence has a binding-c-command.
        """
        id_map = {t["id"]: t for t in sent if isinstance(t["id"], int)}

        for tok in id_map.values():
            feats = tok["feats"] or {}
            if feats.get("Reflex") == "Yes":
                return True

class InterrogativeWhModifierFilter(CorpusFilter):
    """
    Direct wh-questions where the wh-word modifies another phrase
    (ADJ/ADV/NOUN/PRON), e.g.:
        "How tall is he?"
        "Which book did you read?"
        "Whose car is that?"
        "Which one did you pick?"
    """

    wh_lemmas = {"how", "whose", "what", "which"}
    n_upos = {"ADJ", "ADV", "NOUN", "PRON"}

    @property
    def name(self) -> str:
        return "interrogative-wh-modifier"

    def _exclude_sent(self, sent: TokenList) -> bool:
        id_map = {t["id"]: t for t in sent if isinstance(t["id"], int)}

        # find the root
        root = next((t for t in id_map.values() if t["deprel"] == "root"), None)
        if root is None:
            return False

        # require a "?" punct child of the root
        has_qmark = any(
            t["head"] == root["id"]
            and t["deprel"] == "punct"
            and "?" in (t["form"] or "")
            for t in id_map.values()
        )
        if not has_qmark:
            return False

        # find dependents N of the root with the right upos
        for n in id_map.values():
            if n["head"] != root["id"]:
                continue
            if n["upos"] not in self.n_upos:
                continue
            # find a dependent W of N that is an interrogative wh-word
            for w in id_map.values():
                if w["head"] != n["id"]:
                    continue
                feats = w["feats"] or {}
                if feats.get("PronType") != "Int":
                    continue
                if (w["lemma"] or "").lower() in self.wh_lemmas:
                    return True
        return False



        



