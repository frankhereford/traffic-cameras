import { z } from "zod";

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc";

export const visionRouter = createTRPCRouter({
  //   processImage: protectedProcedure
  //     .input(
  //       z.object({
  //         image: z.string(),
  //       }),
  //     )
  //     .mutation(async ({ ctx, input }) => {
  //       console.log("input", input);
  //     }),
});
