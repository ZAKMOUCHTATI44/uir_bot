import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { JSONLoader } from "langchain/document_loaders/fs/json";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { getVectoreStore } from "./vector";
const main = async () => {
  const links = [
    "https://snazzy-brioche-80dbb5.netlify.app/",
    "https://snazzy-brioche-80dbb5.netlify.app/dossier-dinscription.html",
    "https://snazzy-brioche-80dbb5.netlify.app/frais-hebergement.html",
    "https://snazzy-brioche-80dbb5.netlify.app/mot-du-president.html",
  ];

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const vectorStore = await getVectoreStore();

  for (const link of links) {
    const cheerioLoader = new CheerioWebBaseLoader(link, { selector: "body" });
    const docs = await cheerioLoader.load();

    const allSplits = await splitter.splitDocuments(docs);
    // Index chunks
    await vectorStore.addDocuments(allSplits);
  }

  const loader = new JSONLoader("./data/faq.json", ["/answer"]);
  const loaderTxt = new TextLoader("./data/general.txt");
  const loaderCsv = new CSVLoader("./data/data.csv");

  const docsfaq = await loader.load();
  const docsCsv = await loaderCsv.load();
  const loaderTxtLoad = await loaderTxt.load();

  await vectorStore.addDocuments(await splitter.splitDocuments(docsfaq));
  await vectorStore.addDocuments(await splitter.splitDocuments(docsCsv));
  await vectorStore.addDocuments(await splitter.splitDocuments(loaderTxtLoad));

  console.log("DONE");
};

main();
