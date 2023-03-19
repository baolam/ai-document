const brain = require("./brain");
const path = require("path");

const observe = new brain();
observe.watchFolder(path.join(__dirname));