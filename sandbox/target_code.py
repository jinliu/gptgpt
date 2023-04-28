def assemble_prompt(
        system_message_template: Union[str, BaseStringMessagePromptTemplate],
        human_message_template: Union[str, BaseStringMessagePromptTemplate],
):
    sys_prompt, human_prompt = system_message_template, human_message_template
    if isinstance(sys_prompt, str):
        sys_prompt = SystemMessagePromptTemplate.from_template(system_message_template)
    if isinstance(human_prompt, str):
        human_prompt = HumanMessagePromptTemplate.from_template(human_message_template)

    prompt = ChatPromptTemplate.from_messages([sys_prompt, human_prompt])

    return prompt

def get_advice_function(
        system_message_template: Union[str, BaseStringMessagePromptTemplate],
        human_message_template: Union[str, BaseStringMessagePromptTemplate],
        output_class: Type[T]
) -> Callable:
    inner_parser = PydanticOutputParser(pydantic_object=output_class)
    prompt = assemble_prompt(system_message_template, human_message_template)

    async def advice_function(resources: TypedDict) -> T:
        llm = get_default_llm()
        chain = LLMChain(llm=llm, prompt=prompt)
        print(prompt.format(format_instruction=inner_parser.get_format_instructions(), **resources))
        result = await chain.arun(format_instruction=inner_parser.get_format_instructions(), **resources)
        print(result)
        output = OutputFixingParser.from_llm(parser=inner_parser, llm=llm).parse(result)
        return output
    return advice_function