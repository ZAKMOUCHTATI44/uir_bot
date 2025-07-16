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

// ✅ Intent Classification Chain
const classifyPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `Vous êtes un classificateur. Déterminez l'intention principale du message de l'utilisateur.
Choisissez une seule catégorie parmi :
- "greeting" (si c'est une salutation)
- "thanks" (si c'est un remerciement ou une fin polie)
- "question" (si c'est une demande d'information)
- "other" (si aucun des cas ci-dessus ne s'applique)
Répondez uniquement par une de ces étiquettes, sans explication.`,
  ],
  ["human", "{input}"],
]);

const classifyChain = RunnableSequence.from([
  classifyPrompt,
  new ChatOpenAI({ temperature: 0, modelName: "gpt-3.5-turbo" }),
  new StringOutputParser(),
]);

// ✅ Main Function
export const mainFunction = async (userInput: string, phoneNumber: string) => {
  // Detect user intent
  const intent = await classifyChain.invoke({ input: userInput });

  if (intent === "greeting") {
    return `Bonjour! Je suis l'assistant virtuel de l'Université Internationale de Rabat. Comment puis-je vous aider aujourd'hui ? Avez-vous des questions sur nos programmes, les admissions ou d'autres services ?`;
  }

  if (intent === "thanks") {
    return "Je reste à votre disposition si vous avez d’autres questions.";
  }

  // Vector store setup
  const vectorStore = await getVectoreStore();
  const retriever = vectorStore.asRetriever();

  // Convert documents to string
  const convertDocsToString = (documents: Document[]): string => {
    const docs = documents
      .map((document) => `<doc>\n${document.pageContent}\n</doc>`)
      .join("\n");
    console.log(docs);
    return docs;
  };

  // Rephrase question prompt
  const REPHRASE_QUESTION_SYSTEM_TEMPLATE = `Vous êtes l'assistant virtuel de l'Université Internationale de Rabat.
Reformulez la question suivante comme une question autonome, sans mentionner la conversation précédente.`;

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

  // Answer generation prompt
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

  // Retrieval chain
  const documentRetrievalChain = RunnableSequence.from([
    (input) => input.standalone_question,
    retriever,
    convertDocsToString,
  ]);

  // Final QA chain
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

  // Postgres message history setup
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
      return chatHistory;
    },
    historyMessagesKey: "history",
    inputMessagesKey: "question",
  });

  // Invoke the full chain
  const finalResult = await finalRetrievalChain.invoke(
    { question: userInput },
    {
      configurable: {
        sessionId: phoneNumber,
      },
    }
  );

  return finalResult;
};
