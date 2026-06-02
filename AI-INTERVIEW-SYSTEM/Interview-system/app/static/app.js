const socket =
new WebSocket(
    `ws://${window.location.host}/ws`
);

const startBtn =
document.getElementById(
    "startBtn"
);

const reaction =
document.getElementById(
    "reaction"
);

const feedback =
document.getElementById(
    "feedback"
);

const question =
document.getElementById(
    "question"
);

const statusText =
document.getElementById(
    "status"
);

const avatar =
document.getElementById(
    "avatar"
);

const mouth =
document.getElementById(
    "mouth"
);


// Speech Recognition


const SpeechRecognition =
window.SpeechRecognition ||
window.webkitSpeechRecognition;

const recognition =
new SpeechRecognition();

recognition.continuous = false;
recognition.interimResults = false;
recognition.lang = "en-US";


// Avatar Animation


function animateTalking() {

    return setInterval(() => {

        mouth.style.height =
            Math.random() * 30 + "px";

    }, 100);
}

function stopTalking(interval) {

    clearInterval(interval);

    mouth.style.height =
        "10px";
}


// AI SPEAK


function speak(
    text,
    callback = null
) {

    const utterance =
    new SpeechSynthesisUtterance(
        text
    );

    utterance.rate = 1;
    utterance.pitch = 1;

    const talking =
    animateTalking();

    utterance.onstart = () => {

        statusText.innerText =
            "AI Speaking...";
    };

    utterance.onend = () => {

        stopTalking(talking);

        statusText.innerText =
            "Listening...";

        if (callback)
            callback();
    };

    speechSynthesis.speak(
        utterance
    );
}


// Start Interview


startBtn.onclick = () => {

    startBtn.disabled = true;

    socket.send(
        JSON.stringify({
            action: "start"
        })
    );
};


// Start Listening


function startListening() {

    statusText.innerText =
        "Listening...";

    recognition.start();
}


// User Speech


recognition.onresult =
(event) => {

    const transcript =
    event.results[0][0]
    .transcript;

    console.log(
        "USER:",
        transcript
    );

    feedback.innerText =
        "Analyzing answer...";

    statusText.innerText =
        "Thinking...";

    socket.send(
        JSON.stringify({
            action: "answer",
            message: transcript
        })
    );
};


// AI Response


socket.onmessage =
(event) => {

    const data =
    JSON.parse(
        event.data
    );

    reaction.innerText =
        data.reaction;

    feedback.innerText =
        data.feedback;

    question.innerText =
        data.next_question;

    const aiSpeech = `
    ${data.reaction}.
    ${data.feedback}.
    ${data.next_question}
    `;

    speak(
        aiSpeech,
        startListening
    );
};

recognition.onerror =
(event) => {

    console.error(
        event.error
    );

    statusText.innerText =
        "Try speaking again";
};

recognition.onend =
() => {

    console.log(
        "Recognition ended"
    );
};