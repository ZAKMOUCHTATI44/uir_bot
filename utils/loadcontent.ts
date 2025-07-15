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
    "https://snazzy-brioche-80dbb5.netlify.app/dossier-dinscription",
    "https://snazzy-brioche-80dbb5.netlify.app/frais-de-scolarite",
    "https://snazzy-brioche-80dbb5.netlify.app/frais-hebergement.html",
    "https://snazzy-brioche-80dbb5.netlify.app/mot-du-president",
    "https://snazzy-brioche-80dbb5.netlify.app/calendrier-des-concours",
    "https://snazzy-brioche-80dbb5.netlify.app/general.html",
    "https://snazzy-brioche-80dbb5.netlify.app/list-offres.html",
    "https://snazzy-brioche-80dbb5.netlify.app/contact.html",
    "https://snazzy-brioche-80dbb5.netlify.app/sport-loisir.html",
    "https://snazzy-brioche-80dbb5.netlify.app/services-annexes.html",
    "https://snazzy-brioche-80dbb5.netlify.app/admission.html",
    "https://brilliant-ganache-418aa1.netlify.app/bourse.html",
    "https://brilliant-ganache-418aa1.netlify.app/master.html",
    "https://brilliant-ganache-418aa1.netlify.app/inscription.html",
  ];

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const vectorStore = await getVectoreStore();

  for (const link of links) {
    const cheerioLoader = new CheerioWebBaseLoader(link, {
      selector: "section",
    });
    const docs = await cheerioLoader.load();
    const allSplits = await splitter.splitDocuments(docs);
    // Index chunks
    await vectorStore.addDocuments(allSplits);
  }
  console.log("DONE");
};

main();
