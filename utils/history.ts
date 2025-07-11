import pg from "pg";
import { PostgresChatMessageHistory } from "@langchain/community/stores/message/postgres";
import { ChatOpenAI } from "@langchain/openai";
import {
  RunnableConfig,
  RunnablePassthrough,
  RunnableSequence,
  RunnableWithMessageHistory,
} from "@langchain/core/runnables";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { llm } from "./openai";
import { Document } from "@langchain/core/documents";
import { getVectoreStore } from "./vector";
import { MessagesAnnotation } from "@langchain/langgraph";

export const getChainWithHistory = async () => {
  const poolConfig = {
    host: process.env.DB_HOST,
    port: Number(process.env.DB_PORT),
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
  };

  const pool = new pg.Pool(poolConfig);

  const vectorStore = await getVectoreStore();
  const retriever = vectorStore.asRetriever();

  const REPHRASE_QUESTION_SYSTEM_TEMPLATE = `Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question.`;

  const convertDocsToString = (documents: Document[]): string => {
    const docs = documents
      .map((document) => `<doc>\n${document.pageContent}\n</doc>`)
      .join("\n");
    console.log(docs);
    return docs;
  };

  const ANSWER_CHAIN_SYSTEM_TEMPLATE = `Vous êtes l'assistant de l'Université Internationale de Rabat. Répondez poliment et professionnellement. 
  Si l'utilisateur vous salue ou bien 
  si la question contient des mots comme 'bonjour', 'hello', etc., 
  répondez avec : \"Bonjour! Je suis l'assistant virtuel de l'Université Internationale de Rabat. Comment puis-je vous aider aujourd'hui ? Avez-vous des questions sur nos programmes, les admissions ou peut-être cherchez-vous des informations générales sur l'université ?\". Si la réponse ne se trouve pas dans le contexte fourni, répondez par : \"Pouvez-vous reformuler cette question ?
<chat_history>
{chat_history}
</chat_history> 

Now, answer this question using the above context:

{question}
`;

  const answerGenerationChainPrompt = ChatPromptTemplate.fromMessages([
    ["system", ANSWER_CHAIN_SYSTEM_TEMPLATE],
    new MessagesPlaceholder("chat_history"),
    ["human", "{question}"],
  ]);

  const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
    ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
    new MessagesPlaceholder("chat_history"),
    [
      "human",
      "Rephrase the following question as a standalone question:\n{question}",
    ],
  ]);

  const rephraseQuestionChain = RunnableSequence.from([
    rephraseQuestionChainPrompt,
    new ChatOpenAI({
      apiKey: process.env.OPENAI_API_KEY,
      temperature: 0.1,
      modelName: "gpt-3.5-turbo-1106",
    }),
    new StringOutputParser(),
  ]);

  const documentRetrievalChain = RunnableSequence.from([
    (input) => {
      console.log(input.standalone_question);
      return input.standalone_question;
    },
    retriever,
    convertDocsToString,
  ]);

  const conversationalRetrievalChain = RunnableSequence.from([
    RunnablePassthrough.assign({
      standalone_question: rephraseQuestionChain,
      question: rephraseQuestionChain,
    }),
    RunnablePassthrough.assign({
      context: documentRetrievalChain,
    }),
    answerGenerationChainPrompt,
    llm,
    new StringOutputParser(),
  ]);

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `Vous êtes l'assistant de l'Université Internationale de Rabat. Répondez poliment et professionnellement. 
  Si l'utilisateur vous salue
répondez avec : \"Bonjour! Je suis l'assistant virtuel de l'Université Internationale de Rabat. Comment puis-je vous aider aujourd'hui ? Avez-vous des questions sur nos programmes, les admissions ou peut-être cherchez-vous des informations générales sur l'université ?\"`,
    ],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
  ]);

  const chain = prompt.pipe(llm).pipe(new StringOutputParser());

  const chainWithHistory = new RunnableWithMessageHistory({
    runnable: chain,
    inputMessagesKey: "input", // Changed to match the prompt
    historyMessagesKey: "chat_history",
    getMessageHistory: async (sessionId) => {
      const chatHistory = new PostgresChatMessageHistory({
        sessionId,
        pool,
      });
      return chatHistory;
    },
  });

  return chainWithHistory;
};
