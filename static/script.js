(() => {
  const $ = (id) => document.getElementById(id);
  const form=$("chatForm"), input=$("questionInput"), send=$("sendButton"), count=$("charCount");
  const welcome=$("welcomeView"), convo=$("conversationView"), messages=$("messages"), typing=$("typingIndicator");
  const status=$("statusText"), clear=$("clearChatBtn"), suggestions=$("suggestionGrid");
  let busy=false, started=false;
  function refresh(){count.textContent=input.value.length+" / 6000";send.disabled=busy||!input.value.trim();input.style.height="auto";input.style.height=Math.min(input.scrollHeight,180)+"px";}
  function reveal(){if(!started){started=true;welcome.classList.add("hidden");convo.classList.remove("hidden");}}
  function add(role,text,sources=[],error=false){
    reveal();const article=document.createElement("article");article.className="message "+(role==="user"?"user-message":"assistant-message");
    const avatar=document.createElement("div");avatar.className="avatar";avatar.textContent=role==="user"?"Y":"A";
    const body=document.createElement("div");body.className="message-body";const label=document.createElement("div");label.className="message-label";label.textContent=role==="user"?"You":"Astra";
    const content=document.createElement("div");content.className="message-content"+(error?" error-text":"");content.textContent=text;body.append(label,content);article.append(avatar,body);
    if(role==="assistant"&&sources.length){const panel=document.createElement("section");panel.className="source-panel";const heading=document.createElement("div");heading.className="source-heading";heading.textContent="Sources from your book";const list=document.createElement("div");list.className="source-list";
      sources.forEach(source=>{const chip=document.createElement("span");chip.className="source-chip";chip.textContent="Page "+(source.page??"Unknown")+(typeof source.score==="number"?" · "+(source.score*100).toFixed(1)+"%":"");list.append(chip);});panel.append(heading,list);body.append(panel);}
    messages.append(article);article.scrollIntoView({behavior:"smooth",block:"end"});
  }
  function setBusy(v){busy=v;input.disabled=v;send.disabled=v||!input.value.trim();typing.classList.toggle("hidden",!v);status.textContent=v?"Thinking…":"Assistant ready";if(!v)input.focus();}
  async function ask(value){const question=value.trim();if(!question||busy)return;add("user",question);input.value="";refresh();setBusy(true);
    try{const res=await fetch("/api/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:question,top_k:3})});const data=await res.json();if(!res.ok)throw new Error(data.detail||"Request failed");add("assistant",data.answer||"No answer received.",Array.isArray(data.sources)?data.sources:[]);}
    catch(err){add("assistant","Sorry, I could not get a response. "+(err.message||"Check the server and try again."),[],true);}
    finally{setBusy(false);refresh();}}
  form.addEventListener("submit",e=>{e.preventDefault();ask(input.value);});input.addEventListener("input",refresh);
  input.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();if(!send.disabled)form.requestSubmit();}});
  suggestions.addEventListener("click",e=>{const b=e.target.closest("[data-prompt]");if(b&&!busy)ask(b.dataset.prompt||"");});
  clear.addEventListener("click",()=>{if(busy)return;messages.replaceChildren();started=false;convo.classList.add("hidden");welcome.classList.remove("hidden");status.textContent="Assistant ready";input.value="";refresh();input.focus();});
  refresh();
})();