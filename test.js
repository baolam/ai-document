const notifier = require("node-notifier");

notifier.notify("Xin chào");
notifier.on("click", () => {
    console.log("Bạn đã click");
});
notifier.on("timeout", () => {
    console.log("Tự mất");
})