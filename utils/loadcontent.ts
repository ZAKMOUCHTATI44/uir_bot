import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { JSONLoader } from "langchain/document_loaders/fs/json";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { getVectoreStore } from "./vector";
import { link } from "fs";
const main = async () => {
  const links = [
    "https://brilliant-ganache-418aa1.netlify.app?v=20",
    "https://brilliant-ganache-418aa1.netlify.app/dossier-dinscription?v=20",
    "https://brilliant-ganache-418aa1.netlify.app/frais-de-scolarite?v=20",
    "https://brilliant-ganache-418aa1.netlify.app/frais-hebergement.html?v=20",
    "https://brilliant-ganache-418aa1.netlify.app/mot-du-president?v=20",
    "https://brilliant-ganache-418aa1.netlify.app/calendrier-des-concours?v=20",
    "https://brilliant-ganache-418aa1.netlify.app/general.html?v=20",
    "https://brilliant-ganache-418aa1.netlify.app/list-offres.html?v=20",
    "https://brilliant-ganache-418aa1.netlify.app/contact.html?v=20",
    "https://brilliant-ganache-418aa1.netlify.app/sport-loisir.html?v=20",
    "https://brilliant-ganache-418aa1.netlify.app/services-annexes.html?v=20",
    "https://brilliant-ganache-418aa1.netlify.app/admission.html?v=20",
    "https://brilliant-ganache-418aa1.netlify.app/bourse.html?v=20",
    "https://brilliant-ganache-418aa1.netlify.app/master.html?v=20",
    "https://brilliant-ganache-418aa1.netlify.app/inscription.html?v=20",
    "https://brilliant-ganache-418aa1.netlify.app/listing-master.html?v=20",
    "https://brilliant-ganache-418aa1.netlify.app/cours-post-bac.html?v=20",
    "https://brilliant-ganache-418aa1.netlify.app/cours-master.html?v=20",
    "https://brilliant-ganache-418aa1.netlify.app/livret-institutionnel.html?v=20",
    'https://brilliant-ganache-418aa1.netlify.app/team.html'
  ];

  console.log(links.length);

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
