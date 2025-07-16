import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  RunnablePassthrough,
  RunnableSequence,
  RunnableWithMessageHistory,
} from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import pg from "pg";
import { Document } from "@langchain/core/documents";
import { getVectoreStore } from "./vector";
import { PostgresChatMessageHistory } from "@langchain/community/stores/message/postgres";
require("dotenv").config();

// Greeting detection setup
const GREETINGS = [
  "bonjour",
  "bonsoir",
  "salut",
  "slt",
  "hello",
  "hi",
  "hey",
  "yoo",
  "salam",
  "slm",
  "salem",
  "salam alaykoum",
  "salam aleykoum",
  "salem alikoum",
  "salam alikoum",
  "azul", // Tamazight (Berber greeting)
  "allo", // Like “hello” on the phone
  "yo", // Informal
  "good morning",
  "good evening",
  "morning",
  "evening",
];

function containsGreeting(text: string): boolean {
  const lowered = text.toLowerCase();
  return GREETINGS.some((greet) => lowered.includes(greet));
}

function isOnlyGreeting(text: string): boolean {
  const lowered = text.toLowerCase().trim();
  return GREETINGS.some((greet) => {
    const regex = new RegExp(`^${greet}[\\s!.,?]*$`, "i");
    return regex.test(lowered);
  });
}

export const mainFunction = async (userInput: string, phoneNumber: string) => {
  // Case 1: Greeting only
  if (isOnlyGreeting(userInput)) {
    return `Bonjour! Je suis l'assistant virtuel de l'Université Internationale de Rabat. Comment puis-je vous aider aujourd'hui ? Avez-vous des questions sur nos programmes, les admissions ou peut-être cherchez-vous des informations générales sur l'université ?`;
  }

  // Set up vector store & retriever
  const vectorStore = await getVectoreStore();
  const retriever = vectorStore.asRetriever();

  // Converts retrieved docs into string
  const convertDocsToString = (documents: Document[]): string => {
    return documents
      .map((document) => `<doc>\n${document.pageContent}\n</doc>`)
      .join("\n");
  };

  // Rephrase user input into standalone question
  const REPHRASE_QUESTION_SYSTEM_TEMPLATE = `Vous êtes l'assistant virtuel de l'Université Internationale de Rabat.

Sinon, reformulez la question comme une question autonome sans mentionner la conversation précédente.`;

  const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
    ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    [
      "human",
      "Rephrase the following question as a standalone question:\n{question}",
    ],
  ]);

  const rephraseQuestionChain = RunnableSequence.from([
    rephraseQuestionChainPrompt,
    new ChatOpenAI({ temperature: 0.1, modelName: "gpt-3.5-turbo-1106" }),
    new StringOutputParser(),
  ]);

  // Answer generation with context
  const ANSWER_CHAIN_SYSTEM_TEMPLATE = `Vous êtes l'assistant virtuel de l'Université Internationale de Rabat. Répondez poliment, professionnellement, et uniquement à partir des informations fournies dans le contexte ci-dessous.

Si vous ne trouvez pas l'information dans le contexte, dites simplement :
"Pouvez-vous reformuler cette question ?"

<context>
{context}
</context>`;

  const answerGenerationChainPrompt = ChatPromptTemplate.fromMessages([
    ["system", ANSWER_CHAIN_SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    [
      "human",
      "Now, answer this question using the previous context and chat history:\n{standalone_question}",
    ],
  ]);

  // Document retrieval chain
  const documentRetrievalChain = RunnableSequence.from([
    (input) => input.standalone_question,
    retriever,
    convertDocsToString,
  ]);

  // Main question-answering chain
  const conversationalRetrievalChain = RunnableSequence.from([
    RunnablePassthrough.assign({
      standalone_question: rephraseQuestionChain,
    }),
    RunnablePassthrough.assign({
      context: documentRetrievalChain,
    }),
    answerGenerationChainPrompt,
    new ChatOpenAI({ modelName: "gpt-3.5-turbo" }),
    new StringOutputParser(),
  ]);

  // Set up Postgres history storage
  const poolConfig = {
    host: process.env.DB_HOST,
    port: Number(process.env.DB_PORT),
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
  };

  const pool = new pg.Pool(poolConfig);

  const finalRetrievalChain = new RunnableWithMessageHistory({
    runnable: conversationalRetrievalChain,
    getMessageHistory: async (sessionId) => {
      const chatHistory = new PostgresChatMessageHistory({
        sessionId,
        pool,
      });

      console.log(chatHistory)
      return chatHistory;
    },
    historyMessagesKey: "history",
    inputMessagesKey: "question",
  });

  // Run the full chain
  const finalResult = await finalRetrievalChain.invoke(
    {
      question: userInput,
    },
    {
      configurable: { sessionId: phoneNumber },
    }
  );

  return finalResult;
};
