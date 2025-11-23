from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
) # type: ignore

model1 = ChatHuggingFace(llm = llm)

model2 = ChatGoogleGenerativeAI(
    model= 'gemini-2.5-flash'
)

prompt1 = PromptTemplate(
    template= 'Generate simple and short note on the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template= 'Generate the 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template= 'Merge the provided notes and quiz into a single document \n notes -> {chain_notes} and quiz -> {chain_quiz}',
    input_variables=['chain_notes', 'chain_quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'chain_notes': prompt1 | model1 |parser,
    'chain_quiz': prompt2| model2| parser
})

merge_chain = prompt3 | model2 | parser

final_chain = parallel_chain | merge_chain 


text = '''
Chandragupta Maurya[d] (reigned c. 320 BCE[e] – c. 298 BCE)[f] was the founder and the first emperor of the Maurya Empire, based in Magadha (present-day Bihar) in the Indian subcontinent.

His rise to power began in the period of unrest and local warfare that arose after Alexander the Great's Indian campaign and early death in 323 BCE, although the exact chronology and sequence of events remains subject to debate among historians. He started a war against the unpopular Nanda dynasty in Magadha on the Ganges Valley,[6] defeated them and established his own dynasty. In addition, he raised an army to resist the Greeks,[7][8][9][g] defeated them, and took control of the eastern Indus Valley.[10] His conquest of Magadha is generally dated to c. 322–319 BCE,[11][12] and his expansion to Punjab subsequently at c. 317–312 BCE,[h] but some scholars have speculated that he might have initially consolidated his power base in Punjab, before conquering Magadha;[h] an alternative chronology places these events all in the period c. 311–305 BCE.[13][e] According to the play Mudrarakshasa, Chandragupta was assisted by his mentor Chanakya, who later became his minister. He expanded his reach subsequently into parts of the western Indus Valley[i] and possibly[14] eastern Afghanistan[b] through a dynastic marriage alliance with Seleucus I Nicator c. 305–303 BCE.[10] His empire also included Gujarat[j] and a geographically extensive network of cities and trade-routes.[a][b]

There are no historical facts about Chandragupta's origins and early life, only legends, while the narrative of his reign is mainly deduced from a few fragments in Greek and Roman sources, and a few Indian religious texts, all written centuries after his death. The prevailing levels of technology and infrastructure limited the extent of Chandragupta's rule,[k] and the administration was decentralised, with provinces and local governments,[15][l] and large autonomous regions within its limits.[m] Chandragupta's reign, and the Maurya Empire, which reached its peak under his grandson Ashoka the Great,[n] began an era of economic prosperity, reforms, infrastructure expansions. Buddhism, Jainism and Ājīvika prevailed over the non-Maghadian Vedic and Brahmanistic traditions,[16] initiating, under Ashoka, the expansion of Buddhism, and the synthesis of Brahmanic and non-Brahmanic religious traditions which converged in Hinduism. His legend still inspires visions of an undivided Indian nation.

Historical sources
Chandragupta's confrontations with the Greeks and the Nanda king are shortly referred to in a few passages in Greek-Roman sources from the 1st century BCE to the 2nd century CE. Impressions of India at that time are given by a number of other Greek sources. He is further mentioned in Brahmanical, Buddhist, and Jain religious texts and legends, which give impressions of his later reception; they significantly vary in detail.[17] According to Mookerji, the main sources on Chandragupta and his time, in chronological order are:[18]

Greek sources by three companions of Alexander, namely Nearchus, Onesicritus, and Aristobulus of Cassandreia, who write about Alexander and do not mention Chandragupta;
The Greek ambassador Megasthanes, a contemporary of Chandragupta, whose works are lost, but fragments are preserved in the works of other authors, namely Greco-Roman authors Strabo (64 BCE–19 CE), Diodorus (died c. 36 BCE, wrote about India), Arrian (c. 130–172 CE, wrote about India), Pliny the Elder (1st cent. CE, wrote about India), Plutarch (c. 45–125 CE), and Justin (2nd cent. CE). According to Mookerji, without these sources this period would be "a most obscure chapter of Indian history."[19]
The Brahmanical Puranas (Gupta-times), religious texts which viewed the Nandas and Mauryas as illegitimate rulers, because of their shudra background;
Later Brahmanical narratives include legends in Vishakhadatta's Mudrarakshasa (4th–8th cent), Somadeva's Kathasaritsagara (11th cent.) and Kshemendra's Brihatkathamanjari (11th ). Mookerji includes the Arthasastra as a source, a text now dated to the 1st–3rd century CE, and attributed to Chanakya during Gupta-times.[20]
The earliest Buddhist sources are dated to the fourth-century CE or after, including the Sri Lankan Pali texts Dipavamsa (Rajavamsa section), Mahavamsa, Mahavamsa tika and Mahabodhivamsa.
7th to 10th century Jain inscriptions at Shravanabelgola; these are disputed by scholars as well as the Svetambara Jain tradition.[21][22] The second Digambara text interpreted to be mentioning the Maurya emperor is dated to about the 10th-century such as in the Brhatkathakosa of Harisena (Jain monk), while the complete Jain legend about Chandragupta is found in the 12th-century Parisishtaparvan by Hemachandra.
The Greek and Roman texts do not mention Chandragupta directly, except for a second-century text written by the Roman historian Justin. They predominantly describe India, or mention the last Nanda emperor, who usurped the throne of the king before him (Curtis, Diodorus, Plutarch).[23] Justin states that Chandragupta was of humble origin, and includes stories of miraculous legends associated with him, such as a wild elephant appearing and submitting itself to him as a ride to him before a battle. Justin's text states that Chandragupta "achieved [India's] freedom, and "aspired to royalty by all men," as he offended Nanda and was ordered to death, but saved himself "by a speedy flight."[24]

Plutarch states that Chandragupta, as a young man, saw Alexander the Great.[25] He is described as a great king, but not as great in power and influence as Porus in northwestern India or Agrammes (Dhana Nanda) in eastern India.[26]

The Brahmanical Puranic texts do not discuss the details of Chandragupta's ancestry, but rather cover the ancestry of the last Nanda king, and the restoration of just rule by Kautilya[27] (Chanakya; the identification with Kautilya, the author of the Arthashastra, dates from a later period[20]). The Nanda king is described to be cruel, against dharma and shastras, and born out of an illicit relationship followed by a coup.[27] According to Mookerji, the Arthasastra refers to the Nanda rule as against the spiritual, cultural, and military interests of the country, a period where intrigue and vice multiplied.[27] In a later addition,[20] the Arthasastra states that the text was written by him who returned dharma, nurtured diversity of views, and ruled virtuously that kindled love among the subjects for his rule,[27] an insertion linking the Guptas to the Mauryans.[20]

Buddhist texts such as Mahavamsa describe Chandragupta to be of Kshatriya origin.[28] These sources, written about seven centuries after his dynasty ended, state that both Chandragupta and his grandson Ashoka – a patron of Buddhism – were Moriyas, a branch of Gautama Buddha's Shakya noble family.[29] These Buddhist sources attempt to link the dynasty of their patron Ashoka directly to the Buddha.[30] The sources claim that the family branched off to escape persecution from a King of Kosala and Chandragupta's ancestors moved into a secluded Himalayan kingdom known for its peacocks. The Buddhist sources explain the epithet maurya comes from these peacocks, or Mora in Pali (Sanskrit: Mayura).[29][1] The Buddhist texts are inconsistent; some offer other legends to explain his epithet. For example, they mention a city named "Moriya-nagara" where all buildings were made of bricks colored like the peacock's neck.[31] The Maha-bodhi-vasa states he hailed from Moriya-nagara, while the Digha-Nikaya states he came from the Maurya clan of Pipphalivana.[28] The Buddhist sources also mention that "Brahmin Chanakya" was his counselor and with whose support Chandragupta became the king at Patliputra.[31] He has also been variously identified with Shashigupta (which has same etymology as of Chandragupta) of Paropamisadae on the account of same life events
'''
result = final_chain.invoke({'text':text})

print(result)

final_chain.get_graph().print_ascii()