        next_chunk = ""
        chunk_buffer = ""
        current_chunk = ""
        iseparator = 0

        for i, idoc in enumerate(loaded_docs):
            # This return a page content as a string

            separator = separators[iseparator]
            next_page = idoc.page_content
            next_chunk += next_page
            next_chunk_token_count = count_tokens(next_chunk)
            print(
                f"Processing page {i+1}next_chunk_token_count {next_chunk_token_count} {separator in next_chunk}"
            )
            if next_chunk_token_count > chunk_size and separator in next_chunk:
                # split the doc A / SEPARATOR / B
                chunk_part_a = next_chunk.split(separator, 1)[
                    0
                ]  # split the chunk at the first occurrence of the separator
                chunk_part_b = next_chunk.split(separator, 1)[1]
                chunk_buffer_token_count = count_tokens(chunk_buffer)
                chunk_part_a_token_count = count_tokens(chunk_part_a)
                chunk_part_b_token_count = count_tokens(chunk_part_b)
                print(
                    f"chunk_part_a_token_count {chunk_part_a_token_count} chunk_part_b_token_count {chunk_part_b_token_count} chunk_buffer_token_count {chunk_buffer_token_count}"
                )
                if chunk_part_a_token_count + chunk_buffer_token_count < chunk_size:
                    chunk_buffer += separator + chunk_part_a
                    continue
                elif chunk_part_a_token_count + chunk_buffer_token_count >= chunk_size:
                    if (
                        chunk_part_a_token_count + chunk_buffer_token_count
                        < 3 / 2 * chunk_size
                    ):
                        final_chunk = chunk_buffer + separator + chunk_part_a
                        final_chunk_token_count = count_tokens(final_chunk)
                        fc_metadata = idoc.metadata.copy()
                        fc_metadata["token_count"] = final_chunk_token_count
                        docs_processed.append(
                            LangchainDocument(
                                page_content=final_chunk, metadata=fc_metadata
                            )
                        )
                        iseparator = 0
                        chunk_buffer = separator + chunk_part_b
                        continue
                    elif (
                        chunk_part_a_token_count
                        + chunk_buffer_token_count
                        + chunk_part_b_token_count
                        < 2 * chunk_size
                    ):
                        final_chunk = (
                            chunk_buffer
                            + separator
                            + chunk_part_a
                            + separator
                            + chunk_part_b
                        )
                        final_chunk_token_count = count_tokens(final_chunk)
                        fc_metadata = idoc.metadata.copy()
                        fc_metadata["token_count"] = final_chunk_token_count
                        docs_processed.append(
                            LangchainDocument(
                                page_content=final_chunk, metadata=fc_metadata
                            )
                        )
                        iseparator = 0
                        chunk_buffer = ""
                        continue
                    else:
                        final_chunk = chunk_buffer + separator + chunk_part_a
                        final_chunk_token_count = count_tokens(final_chunk)
                        fc_metadata = idoc.metadata.copy()
                        fc_metadata["token_count"] = final_chunk_token_count
                        docs_processed.append(
                            LangchainDocument(
                                page_content=final_chunk, metadata=fc_metadata
                            )
                        )
                        chunk_buffer = separator + chunk_part_b
                        if len(chunk_buffer) == 0:
                            print("WARNING: B part is big {chunk_part_b_token_count}")
                        iseparator = 0
                        continue

            elif next_chunk_token_count > chunk_size:
                iseparator += 1
            else:
                continue