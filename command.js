const cmd = require("node-cmd");

cmd.run("git add .", (err, data, stderr) => {
    console.log(data);
});